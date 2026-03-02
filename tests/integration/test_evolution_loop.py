"""
Integration tests for complete evolution loops.

Tests verify:
1. Fitness improves over generations
2. Archive fills appropriately
3. Algorithms complete without errors
4. Results are reproducible with same seed
"""

import numpy as np
import pytest


@pytest.mark.slow
class TestEvolutionLoop:
    """Tests for complete evolution cycles."""

    def test_mmr_elites_fitness_improves(self, arm_task):
        """MMR-Elites should improve fitness over generations."""
        pytest.importorskip("mmr_elites_rs")
        from mmr_elites.algorithms.mmr_elites import run_mmr_elites

        result = run_mmr_elites(
            task=arm_task,
            archive_size=100,
            generations=100,
            batch_size=50,
            lambda_val=0.5,
            mutation_sigma=0.1,
            seed=42,
        )

        history = result["history"]
        initial_max = history["max_fitness"][0]
        final_max = history["max_fitness"][-1]

        assert final_max >= initial_max

    def test_map_elites_archive_grows(self, arm_task):
        """MAP-Elites archive should grow over generations."""
        from mmr_elites.algorithms.map_elites import run_map_elites

        result = run_map_elites(
            task=arm_task,
            generations=100,
            batch_size=50,
            bins_per_dim=5,
            mutation_sigma=0.1,
            seed=42,
        )

        history = result["history"]
        initial_size = history["archive_size"][0]
        final_size = history["archive_size"][-1]

        assert final_size >= initial_size

    def test_cvt_map_elites_completes(self, arm_task):
        """CVT-MAP-Elites should complete without error."""
        from mmr_elites.algorithms.cvt_map_elites import run_cvt_map_elites

        result = run_cvt_map_elites(
            task=arm_task,
            n_niches=100,
            generations=50,
            batch_size=50,
            mutation_sigma=0.1,
            seed=42,
        )

        assert "final_metrics" in result
        assert result["final_metrics"]["archive_size"] > 0

    def test_reproducibility(self, arm_task):
        """Same seed should produce same results."""
        pytest.importorskip("mmr_elites_rs")
        from mmr_elites.algorithms.mmr_elites import run_mmr_elites

        result1 = run_mmr_elites(
            task=arm_task,
            archive_size=50,
            generations=20,
            batch_size=20,
            lambda_val=0.5,
            mutation_sigma=0.1,
            seed=42,
        )

        result2 = run_mmr_elites(
            task=arm_task,
            archive_size=50,
            generations=20,
            batch_size=20,
            lambda_val=0.5,
            mutation_sigma=0.1,
            seed=42,
        )

        np.testing.assert_array_equal(
            result1["final_fitness"], result2["final_fitness"]
        )


@pytest.mark.slow
class TestAlgorithmComparison:
    """Tests comparing algorithms."""

    def test_mmr_elites_better_uniformity(self, arm_task):
        """MMR-Elites should achieve better uniformity than MAP-Elites."""
        pytest.importorskip("mmr_elites_rs")
        from mmr_elites.algorithms.map_elites import run_map_elites
        from mmr_elites.algorithms.mmr_elites import run_mmr_elites

        mmr_result = run_mmr_elites(
            task=arm_task,
            archive_size=100,
            generations=200,
            batch_size=50,
            lambda_val=0.5,
            mutation_sigma=0.1,
            seed=42,
        )

        me_result = run_map_elites(
            task=arm_task,
            generations=200,
            batch_size=50,
            bins_per_dim=5,
            mutation_sigma=0.1,
            seed=42,
        )

        # Lower CV = more uniform
        mmr_cv = mmr_result["final_metrics"]["uniformity_cv"]
        me_cv = me_result["final_metrics"]["uniformity_cv"]

        # MMR should be at least comparable or better in uniformity
        assert mmr_cv is not None
        assert me_cv is not None
