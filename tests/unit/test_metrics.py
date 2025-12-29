"""
Unit tests for QD metrics calculations.

Tests verify:
1. QD-Score calculations
2. Coverage and uniformity metrics
3. Fair comparison metrics (qd_score_at_budget)
4. Edge cases
"""

import pytest
import numpy as np
from mmr_elites.metrics.qd_metrics import (
    qd_score,
    qd_score_at_budget,
    max_fitness,
    mean_fitness,
    mean_pairwise_distance,
    archive_uniformity,
    compute_all_metrics,
)


class TestQDScore:
    """Tests for basic QD-Score calculation."""
    
    def test_sum_of_fitness(self):
        """QD-Score is sum of all fitness values."""
        fitness = np.array([0.1, 0.5, 0.9])
        assert qd_score(fitness) == pytest.approx(1.5)
    
    def test_empty_array(self):
        """Empty fitness array returns 0."""
        fitness = np.array([])
        assert qd_score(fitness) == 0.0
    
    def test_single_element(self):
        """Single fitness value returns that value."""
        fitness = np.array([0.75])
        assert qd_score(fitness) == pytest.approx(0.75)


class TestQDScoreAtBudget:
    """Tests for budget-constrained QD-Score (fair comparison metric)."""
    
    def test_sum_top_k(self):
        """Should sum top-K fitness values."""
        fitness = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        # Top 3: 0.9, 0.7, 0.5 = 2.1
        assert qd_score_at_budget(fitness, 3) == pytest.approx(2.1)
    
    def test_budget_exceeds_size(self):
        """When budget > len(fitness), sum all."""
        fitness = np.array([0.1, 0.2, 0.3])
        assert qd_score_at_budget(fitness, 10) == pytest.approx(0.6)
    
    def test_budget_equals_size(self):
        """When budget == len(fitness), sum all."""
        fitness = np.array([0.4, 0.5, 0.6])
        assert qd_score_at_budget(fitness, 3) == pytest.approx(1.5)
    
    def test_budget_one(self):
        """Budget of 1 returns max fitness."""
        fitness = np.array([0.1, 0.9, 0.5])
        assert qd_score_at_budget(fitness, 1) == pytest.approx(0.9)
    
    def test_empty_array(self):
        """Empty array returns 0."""
        fitness = np.array([])
        assert qd_score_at_budget(fitness, 10) == 0.0


class TestMeanPairwiseDistance:
    """Tests for diversity metric."""
    
    def test_known_distances(self):
        """Test with known point configuration."""
        # Square: (0,0), (1,0), (0,1), (1,1)
        # Distances: 1, 1, 1, 1, sqrt(2), sqrt(2)
        # Mean = (4 + 2*sqrt(2)) / 6
        descriptors = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        expected = (4 + 2 * np.sqrt(2)) / 6
        assert mean_pairwise_distance(descriptors) == pytest.approx(expected)
    
    def test_single_point(self):
        """Single point returns 0."""
        descriptors = np.array([[0.5, 0.5]])
        assert mean_pairwise_distance(descriptors) == 0.0
    
    def test_two_points(self):
        """Two points returns distance between them."""
        descriptors = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert mean_pairwise_distance(descriptors) == pytest.approx(5.0)
    
    def test_identical_points(self):
        """Identical points returns 0."""
        descriptors = np.ones((10, 5))
        assert mean_pairwise_distance(descriptors) == pytest.approx(0.0)


class TestArchiveUniformity:
    """Tests for uniformity (CV of k-NN distances)."""
    
    def test_uniform_grid_low_cv(self):
        """Uniform grid should have low CV."""
        # Create 10x10 uniform grid
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        descriptors = np.column_stack([xx.ravel(), yy.ravel()])
        
        cv = archive_uniformity(descriptors, k=5)
        assert cv < 0.2  # Low CV = uniform
    
    def test_clustered_high_cv(self):
        """Clustered points should have high CV."""
        np.random.seed(42)
        # Two tight clusters
        cluster1 = np.random.randn(50, 2) * 0.1
        cluster2 = np.random.randn(50, 2) * 0.1 + 10
        descriptors = np.vstack([cluster1, cluster2])
        
        cv = archive_uniformity(descriptors, k=5)
        assert cv > 0.4  # High CV = non-uniform
    
    def test_insufficient_points(self):
        """When n <= k, returns 0."""
        descriptors = np.random.randn(3, 5)
        assert archive_uniformity(descriptors, k=5) == 0.0


class TestComputeAllMetrics:
    """Tests for the combined metrics function."""
    
    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        np.random.seed(42)
        fitness = np.random.rand(100)
        descriptors = np.random.rand(100, 5)
        
        metrics = compute_all_metrics(fitness, descriptors, budget_k=100)
        
        expected_keys = {
            "qd_score", "qd_score_at_budget", "max_fitness", "mean_fitness",
            "mean_pairwise_distance", "uniformity_cv", "archive_size",
            "coverage", "mean_fitness_at_budget"
        }
        # Note: I changed coverage_efficiency to coverage in my implementation for consistency with old code
        assert set(metrics.keys()) == expected_keys
    
    def test_values_consistent(self):
        """Values should match individual function calls."""
        np.random.seed(42)
        fitness = np.random.rand(100)
        descriptors = np.random.rand(100, 5)
        
        metrics = compute_all_metrics(fitness, descriptors, budget_k=50)
        
        assert metrics["qd_score"] == pytest.approx(qd_score(fitness))
        assert metrics["max_fitness"] == pytest.approx(max_fitness(fitness))
        assert metrics["archive_size"] == 100
