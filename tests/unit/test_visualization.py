"""Tests for visualization module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mmr_elites.utils.visualization import (
    COLORS,
    LINESTYLES,
    plot_arm_configurations,
    plot_behavior_space_comparison,
    plot_final_metrics_bars,
    plot_learning_curves,
    save_figure,
    set_publication_style,
)


@pytest.fixture
def mock_results_objects():
    """Create mock result objects with history and final_metrics."""
    results = {}
    for alg_name in ["mmr_elites", "map_elites"]:
        runs = []
        for _ in range(3):
            r = MagicMock()
            r.history = {
                "generation": list(range(0, 100, 10)),
                "qd_score": list(np.random.rand(10).cumsum()),
                "max_fitness": list(np.random.rand(10)),
            }
            r.final_metrics = {
                "qd_score": float(np.random.rand()),
                "max_fitness": float(np.random.rand()),
                "mean_pairwise_distance": float(np.random.rand()),
            }
            runs.append(r)
        results[alg_name] = runs
    return results


@pytest.fixture
def mock_results_dicts():
    """Create mock result dicts with history and final_metrics."""
    results = {}
    for alg_name in ["mmr_elites", "map_elites"]:
        runs = []
        for _ in range(3):
            r = {
                "history": {
                    "generation": list(range(0, 100, 10)),
                    "qd_score": list(np.random.rand(10).cumsum()),
                    "max_fitness": list(np.random.rand(10)),
                },
                "final_metrics": {
                    "qd_score": float(np.random.rand()),
                    "max_fitness": float(np.random.rand()),
                    "mean_pairwise_distance": float(np.random.rand()),
                },
            }
            runs.append(r)
        results[alg_name] = runs
    return results


class TestStyleConfig:
    def test_colors_defined(self):
        assert "mmr_elites" in COLORS
        assert "map_elites" in COLORS
        assert "cvt_map_elites" in COLORS
        assert "random" in COLORS

    def test_linestyles_defined(self):
        assert "mmr_elites" in LINESTYLES
        assert "map_elites" in LINESTYLES

    def test_set_publication_style(self):
        set_publication_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False


class TestPlotLearningCurves:
    def test_basic_plot_with_objects(self, mock_results_objects):
        ax = plot_learning_curves(mock_results_objects, metric="qd_score")
        assert ax is not None
        assert len(ax.lines) > 0
        plt.close("all")

    def test_basic_plot_with_dicts(self, mock_results_dicts):
        ax = plot_learning_curves(mock_results_dicts, metric="qd_score")
        assert ax is not None
        plt.close("all")

    def test_with_existing_axes(self, mock_results_objects):
        fig, ax = plt.subplots()
        returned_ax = plot_learning_curves(
            mock_results_objects, metric="qd_score", ax=ax
        )
        assert returned_ax is ax
        plt.close("all")

    def test_with_title_and_ylabel(self, mock_results_objects):
        ax = plot_learning_curves(
            mock_results_objects,
            metric="qd_score",
            title="Test Title",
            ylabel="Test Label",
        )
        assert ax.get_title() == "Test Title"
        assert ax.get_ylabel() == "Test Label"
        plt.close("all")

    def test_no_smoothing(self, mock_results_objects):
        ax = plot_learning_curves(
            mock_results_objects, metric="qd_score", smooth_sigma=0
        )
        assert ax is not None
        plt.close("all")

    def test_show_individual_runs(self, mock_results_objects):
        ax = plot_learning_curves(
            mock_results_objects, metric="qd_score", show_individual=True
        )
        assert ax is not None
        plt.close("all")

    def test_show_individual_no_smoothing(self, mock_results_objects):
        ax = plot_learning_curves(
            mock_results_objects,
            metric="qd_score",
            show_individual=True,
            smooth_sigma=0,
        )
        assert ax is not None
        plt.close("all")

    def test_unknown_algorithm_uses_gray(self):
        results = {
            "unknown_algo": [
                MagicMock(
                    history={
                        "generation": [0, 10],
                        "qd_score": [1.0, 2.0],
                    }
                )
            ]
        }
        ax = plot_learning_curves(results, metric="qd_score")
        assert ax is not None
        plt.close("all")


class TestPlotFinalMetricsBars:
    def test_basic_bar_chart(self, mock_results_objects):
        fig = plot_final_metrics_bars(mock_results_objects)
        assert fig is not None
        plt.close("all")

    def test_custom_metrics(self, mock_results_objects):
        fig = plot_final_metrics_bars(mock_results_objects, metrics=["qd_score"])
        assert fig is not None
        plt.close("all")

    def test_single_metric(self, mock_results_objects):
        fig = plot_final_metrics_bars(mock_results_objects, metrics=["max_fitness"])
        assert fig is not None
        plt.close("all")

    def test_with_dict_results(self, mock_results_dicts):
        fig = plot_final_metrics_bars(mock_results_dicts)
        assert fig is not None
        plt.close("all")

    def test_custom_figsize(self, mock_results_objects):
        fig = plot_final_metrics_bars(mock_results_objects, figsize=(8, 3))
        assert fig is not None
        plt.close("all")


class TestPlotBehaviorSpaceComparison:
    def test_basic_comparison(self):
        results = {
            "mmr_elites": np.random.rand(50, 2),
            "map_elites": np.random.rand(50, 2),
        }
        fig = plot_behavior_space_comparison(results)
        assert fig is not None
        plt.close("all")

    def test_single_algorithm(self):
        results = {"mmr_elites": np.random.rand(50, 2)}
        fig = plot_behavior_space_comparison(results)
        assert fig is not None
        plt.close("all")

    def test_with_bounds(self):
        results = {"mmr_elites": np.random.rand(50, 2)}
        fig = plot_behavior_space_comparison(results, bounds=(0, 1, 0, 1))
        assert fig is not None
        plt.close("all")

    def test_with_titles(self):
        results = {"mmr_elites": np.random.rand(50, 2)}
        fig = plot_behavior_space_comparison(
            results, titles={"mmr_elites": "MMR-Elites Archive"}
        )
        assert fig is not None
        plt.close("all")

    def test_empty_descriptors(self):
        results = {"mmr_elites": np.empty((0, 2))}
        fig = plot_behavior_space_comparison(results)
        assert fig is not None
        plt.close("all")

    def test_no_titles(self):
        results = {"mmr_elites": np.random.rand(50, 2)}
        fig = plot_behavior_space_comparison(results, titles=None)
        assert fig is not None
        plt.close("all")


class TestPlotArmConfigurations:
    def test_basic_arm_plot(self):
        genomes = np.random.uniform(-np.pi, np.pi, (20, 5))
        fig = plot_arm_configurations(genomes)
        assert fig is not None
        plt.close("all")

    def test_with_sampling(self):
        genomes = np.random.uniform(-np.pi, np.pi, (100, 10))
        fig = plot_arm_configurations(genomes, n_samples=5)
        assert fig is not None
        plt.close("all")

    def test_fewer_than_n_samples(self):
        genomes = np.random.uniform(-np.pi, np.pi, (3, 5))
        fig = plot_arm_configurations(genomes, n_samples=10)
        assert fig is not None
        plt.close("all")

    def test_empty_genomes(self):
        genomes = np.empty((0, 5))
        fig = plot_arm_configurations(genomes)
        assert fig is not None
        plt.close("all")

    def test_no_obstacle(self):
        genomes = np.random.uniform(-np.pi, np.pi, (5, 5))
        fig = plot_arm_configurations(genomes, obstacle_box=None)
        assert fig is not None
        plt.close("all")

    def test_custom_target(self):
        genomes = np.random.uniform(-np.pi, np.pi, (5, 5))
        fig = plot_arm_configurations(genomes, target_pos=(0.5, 0.5))
        assert fig is not None
        plt.close("all")


class TestSaveFigure:
    def test_save_default_formats(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_fig"
            save_figure(fig, str(path))
            assert (Path(f"{path}.pdf")).exists()
            assert (Path(f"{path}.png")).exists()
        plt.close("all")

    def test_save_single_format(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_fig"
            save_figure(fig, str(path), formats=["png"])
            assert (Path(f"{path}.png")).exists()
            assert not (Path(f"{path}.pdf")).exists()
        plt.close("all")

    def test_save_creates_directories(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test_fig"
            save_figure(fig, str(path), formats=["png"])
            assert (Path(f"{path}.png")).exists()
        plt.close("all")
