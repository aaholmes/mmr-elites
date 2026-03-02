"""
Unit tests for CVT-MAP-Elites implementation.
"""

import numpy as np
import pytest


class TestCVTCentroids:
    """Tests for CVT centroid computation."""

    def test_correct_number_of_centroids(self):
        """Should create exactly n_niches centroids."""
        from mmr_elites.algorithms.base import ExperimentConfig
        from mmr_elites.algorithms.cvt_map_elites import CVTMAPElites

        config = ExperimentConfig(archive_size=100)
        cvt = CVTMAPElites(config, n_niches=100)

        centroids = cvt._compute_centroids(desc_dim=5, seed=42)

        assert centroids.shape == (100, 5)

    def test_centroids_in_bounds(self):
        """Centroids should be in [0, 1]^D."""
        from mmr_elites.algorithms.base import ExperimentConfig
        from mmr_elites.algorithms.cvt_map_elites import CVTMAPElites

        config = ExperimentConfig(archive_size=50)
        cvt = CVTMAPElites(config, n_niches=50)

        centroids = cvt._compute_centroids(desc_dim=10, seed=42)

        assert np.all(centroids >= 0)
        assert np.all(centroids <= 1)

    def test_deterministic_centroids(self):
        """Same seed should produce same centroids."""
        from mmr_elites.algorithms.base import ExperimentConfig
        from mmr_elites.algorithms.cvt_map_elites import CVTMAPElites

        config = ExperimentConfig(archive_size=50)
        cvt1 = CVTMAPElites(config, n_niches=50)
        cvt2 = CVTMAPElites(config, n_niches=50)

        c1 = cvt1._compute_centroids(desc_dim=5, seed=42)
        c2 = cvt2._compute_centroids(desc_dim=5, seed=42)

        np.testing.assert_allclose(c1, c2, atol=1e-10)


class TestCVTArchive:
    """Tests for CVT archive operations."""

    def test_niche_assignment(self):
        """Points should be assigned to nearest centroid."""
        from mmr_elites.algorithms.base import ExperimentConfig
        from mmr_elites.algorithms.cvt_map_elites import CVTMAPElites
        from mmr_elites.tasks.arm import ArmTask

        config = ExperimentConfig(archive_size=10, generations=1)
        cvt = CVTMAPElites(config, n_niches=10)

        task = ArmTask(n_dof=5)
        cvt.initialize(task, seed=42)

        # A point exactly at a centroid should map to that niche
        test_point = cvt.centroids[0].copy()
        niche = cvt._get_niche(test_point)
        assert niche == 0

    def test_archive_update_keeps_best(self):
        """Archive should keep best fitness per niche."""
        from mmr_elites.algorithms.base import ExperimentConfig
        from mmr_elites.algorithms.cvt_map_elites import CVTMAPElites
        from mmr_elites.tasks.arm import ArmTask

        config = ExperimentConfig(archive_size=10, generations=1)
        cvt = CVTMAPElites(config, n_niches=10)

        task = ArmTask(n_dof=5)
        cvt.initialize(task, seed=42)

        # Clear archive to test specific additions
        cvt.archive = {}

        # Add a solution
        genome1 = np.zeros(5)
        desc1 = cvt.centroids[0].copy()
        cvt._add_to_archive(genome1, 0.5, desc1)

        niche = cvt._get_niche(desc1)
        assert cvt.archive[niche][1] == 0.5

        # Add better solution to same niche
        genome2 = np.ones(5)
        cvt._add_to_archive(genome2, 0.8, desc1)
        assert cvt.archive[niche][1] == 0.8

        # Worse solution should not replace
        genome3 = np.ones(5) * 2
        cvt._add_to_archive(genome3, 0.3, desc1)
        assert cvt.archive[niche][1] == 0.8


class TestCVTFunctionalInterface:
    """Tests for run_cvt_map_elites function."""

    def test_returns_expected_keys(self):
        """Result should have all expected keys."""
        from mmr_elites.algorithms.cvt_map_elites import run_cvt_map_elites
        from mmr_elites.tasks.arm import ArmTask

        task = ArmTask(n_dof=5)
        result = run_cvt_map_elites(
            task, n_niches=50, generations=10, batch_size=20, seed=42
        )

        expected_keys = {
            "algorithm",
            "seed",
            "runtime",
            "final_metrics",
            "history",
            "final_genomes",
            "final_fitness",
            "final_descriptors",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_archive_grows(self):
        """Archive size should increase over generations."""
        from mmr_elites.algorithms.cvt_map_elites import run_cvt_map_elites
        from mmr_elites.tasks.arm import ArmTask

        task = ArmTask(n_dof=5)
        result = run_cvt_map_elites(
            task, n_niches=100, generations=100, batch_size=50, seed=42, log_interval=10
        )

        sizes = result["history"]["archive_size"]
        assert sizes[-1] >= sizes[0]
