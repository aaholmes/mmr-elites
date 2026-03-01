"""Tests for statistics module."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from mmr_elites.utils.statistics import (
    compute_confidence_interval,
    wilcoxon_signed_rank_test,
    mann_whitney_u_test,
    cohens_d,
    compute_all_statistics,
    format_results_table,
)


class TestComputeConfidenceInterval:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, ci_low, ci_high = compute_confidence_interval(data)
        assert mean == pytest.approx(3.0)
        assert ci_low < mean
        assert ci_high > mean

    def test_single_value(self):
        data = np.array([5.0])
        mean, ci_low, ci_high = compute_confidence_interval(data)
        assert mean == 5.0
        assert ci_low == 5.0
        assert ci_high == 5.0

    def test_custom_confidence(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_95, low_95, high_95 = compute_confidence_interval(data, confidence=0.95)
        mean_99, low_99, high_99 = compute_confidence_interval(data, confidence=0.99)
        # 99% CI should be wider
        assert (high_99 - low_99) > (high_95 - low_95)

    def test_identical_values(self):
        data = np.array([3.0, 3.0, 3.0])
        mean, ci_low, ci_high = compute_confidence_interval(data)
        assert mean == pytest.approx(3.0)


class TestWilcoxonSignedRankTest:
    def test_identical_distributions(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Identical paired samples: stat should be 0 or p should be 1
        stat, p = wilcoxon_signed_rank_test(x, y)
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_different_distributions(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, p = wilcoxon_signed_rank_test(x, y)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_greater_alternative(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, p = wilcoxon_signed_rank_test(x, y, alternative="greater")
        assert p < 0.05  # x is clearly greater


class TestMannWhitneyUTest:
    def test_different_distributions(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, p = mann_whitney_u_test(x, y)
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_similar_distributions(self):
        np.random.seed(42)
        x = np.random.randn(20)
        y = np.random.randn(20)
        stat, p = mann_whitney_u_test(x, y)
        assert p > 0.01  # Should not be significant

    def test_greater_alternative(self):
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([1.0, 2.0, 3.0])
        stat, p = mann_whitney_u_test(x, y, alternative="greater")
        assert p < 0.1


class TestCohensD:
    def test_no_effect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(x, y)
        assert d == pytest.approx(0.0)

    def test_large_effect(self):
        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(x, y)
        assert abs(d) >= 0.8  # Large effect

    def test_zero_variance(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([5.0, 5.0, 5.0])
        d = cohens_d(x, y)
        assert d == 0.0

    def test_sign(self):
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([1.0, 2.0, 3.0])
        d = cohens_d(x, y)
        assert d > 0  # x > y means positive d


class TestComputeAllStatistics:
    @pytest.fixture
    def mock_results(self):
        results = {}
        for alg in ["MMR-Elites", "MAP-Elites", "Random"]:
            runs = []
            base_score = {"MMR-Elites": 100, "MAP-Elites": 80, "Random": 50}[alg]
            for i in range(5):
                r = MagicMock()
                r.final_metrics = {
                    "qd_score_at_budget": base_score + i,
                    "qd_score": base_score + i + 10,
                    "mean_fitness": 0.5 + i * 0.01,
                }
                runs.append(r)
            results[alg] = runs
        return results

    def test_basic_statistics(self, mock_results):
        stats = compute_all_statistics(mock_results)
        assert "MMR-Elites" in stats
        assert "MAP-Elites" in stats
        assert "Random" in stats
        assert "mean" in stats["MMR-Elites"]
        assert "std" in stats["MMR-Elites"]

    def test_confidence_intervals(self, mock_results):
        stats = compute_all_statistics(mock_results)
        for alg in ["MMR-Elites", "MAP-Elites", "Random"]:
            assert stats[alg]["ci_95_low"] <= stats[alg]["mean"]
            assert stats[alg]["ci_95_high"] >= stats[alg]["mean"]

    def test_comparisons_against_baseline(self, mock_results):
        stats = compute_all_statistics(mock_results, baseline="Random")
        assert "_comparisons" in stats
        comps = stats["_comparisons"]
        assert "MMR-Elites" in comps
        assert "MAP-Elites" in comps
        assert "Random" not in comps  # Baseline not compared to itself
        assert "cohens_d" in comps["MMR-Elites"]
        assert "p_value" in comps["MMR-Elites"]

    def test_custom_metric(self, mock_results):
        stats = compute_all_statistics(mock_results, metric="qd_score")
        assert stats["MMR-Elites"]["mean"] > 0

    def test_empty_runs(self):
        results = {"Alg1": [], "Random": []}
        stats = compute_all_statistics(results)
        assert "Alg1" not in stats or len(stats) == 0

    def test_missing_baseline(self):
        results = {
            "Alg1": [
                MagicMock(
                    final_metrics={"qd_score_at_budget": 10, "qd_score": 10}
                )
            ]
        }
        stats = compute_all_statistics(results, baseline="NonExistent")
        assert "_comparisons" not in stats

    def test_dict_results(self):
        results = {
            "Alg1": [
                {"final_metrics": {"qd_score_at_budget": 10, "qd_score": 10}},
                {"final_metrics": {"qd_score_at_budget": 12, "qd_score": 12}},
            ],
            "Random": [
                {"final_metrics": {"qd_score_at_budget": 5, "qd_score": 5}},
                {"final_metrics": {"qd_score_at_budget": 6, "qd_score": 6}},
            ],
        }
        stats = compute_all_statistics(results)
        assert "Alg1" in stats

    def test_identical_values_comparison(self):
        """Test that identical values are handled (ValueError in mann_whitney)."""
        results = {
            "Alg1": [
                MagicMock(final_metrics={"qd_score_at_budget": 10, "qd_score": 10}),
                MagicMock(final_metrics={"qd_score_at_budget": 10, "qd_score": 10}),
            ],
            "Random": [
                MagicMock(final_metrics={"qd_score_at_budget": 10, "qd_score": 10}),
                MagicMock(final_metrics={"qd_score_at_budget": 10, "qd_score": 10}),
            ],
        }
        stats = compute_all_statistics(results, baseline="Random")
        assert "_comparisons" in stats


class TestFormatResultsTable:
    def test_basic_table(self):
        results = {
            "MMR-Elites": [
                MagicMock(
                    final_metrics={
                        "qd_score_at_budget": 100,
                        "mean_fitness": 0.5,
                        "uniformity_cv": 0.1,
                    }
                )
            ],
            "Random": [
                MagicMock(
                    final_metrics={
                        "qd_score_at_budget": 50,
                        "mean_fitness": 0.3,
                        "uniformity_cv": 0.5,
                    }
                )
            ],
        }
        table = format_results_table(results)
        assert "MMR-Elites" in table
        assert "Random" in table

    def test_custom_metrics(self):
        results = {
            "Alg1": [
                MagicMock(final_metrics={"qd_score": 100, "max_fitness": 0.9})
            ],
        }
        table = format_results_table(results, metrics=["qd_score", "max_fitness"])
        assert "qd_score" in table
        assert "max_fitness" in table

    def test_empty_runs_skipped(self):
        results = {"Alg1": [], "Alg2": [MagicMock(final_metrics={"qd_score_at_budget": 10, "mean_fitness": 0.5, "uniformity_cv": 0.1})]}
        table = format_results_table(results)
        assert "Alg1" not in table
        assert "Alg2" in table

    def test_dict_results(self):
        results = {
            "Alg1": [
                {"final_metrics": {"qd_score_at_budget": 10, "mean_fitness": 0.5, "uniformity_cv": 0.1}}
            ]
        }
        table = format_results_table(results)
        assert "Alg1" in table
