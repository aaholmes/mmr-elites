"""
Unit Tests for QD Metrics Module
=================================

Tests correctness of metric calculations against known values.
"""

import numpy as np
import pytest
from mmr_qd.metrics import (
    qd_score,
    max_fitness,
    mean_fitness,
    mean_pairwise_distance,
    archive_uniformity,
    archive_coverage,
    epsilon_coverage,
    sum_of_knn_distances,
    compute_all_metrics,
    aggregate_runs,
)


class TestBasicMetrics:
    """Tests for simple aggregation metrics."""
    
    def test_qd_score_simple(self):
        fitness = np.array([1.0, 2.0, 3.0, 4.0])
        assert qd_score(fitness) == 10.0
    
    def test_qd_score_negative(self):
        fitness = np.array([-1.0, -2.0, 3.0])
        assert qd_score(fitness) == 0.0
    
    def test_max_fitness(self):
        fitness = np.array([1.0, 5.0, 3.0, 2.0])
        assert max_fitness(fitness) == 5.0
    
    def test_mean_fitness(self):
        fitness = np.array([1.0, 2.0, 3.0, 4.0])
        assert mean_fitness(fitness) == 2.5
    
    def test_empty_arrays(self):
        """Metrics should handle edge cases gracefully."""
        empty = np.array([])
        # These should not raise exceptions
        with pytest.raises(ValueError):
            qd_score(empty)


class TestPairwiseDistance:
    """Tests for mean pairwise distance calculation."""
    
    def test_two_points(self):
        # Two points 1 unit apart
        descriptors = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert mean_pairwise_distance(descriptors) == 1.0
    
    def test_three_points_equilateral(self):
        # Equilateral triangle with side 1
        s = 1.0
        descriptors = np.array([
            [0.0, 0.0],
            [s, 0.0],
            [s/2, s * np.sqrt(3)/2]
        ])
        # All pairwise distances are 1
        assert np.isclose(mean_pairwise_distance(descriptors), 1.0, atol=1e-10)
    
    def test_single_point(self):
        descriptors = np.array([[1.0, 2.0]])
        assert mean_pairwise_distance(descriptors) == 0.0
    
    def test_high_dimensional(self):
        # 100 points in 20D should work
        np.random.seed(42)
        descriptors = np.random.randn(100, 20)
        result = mean_pairwise_distance(descriptors)
        assert result > 0
        assert np.isfinite(result)


class TestUniformity:
    """Tests for archive uniformity (coefficient of variation of k-NN distances)."""
    
    def test_perfectly_uniform_grid(self):
        # Regular 10x10 grid should have low CV
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        descriptors = np.column_stack([xx.ravel(), yy.ravel()])
        
        cv = archive_uniformity(descriptors, k=4)
        # Grid has very uniform spacing, CV should be low
        assert cv < 0.5
    
    def test_clustered_distribution(self):
        # Two tight clusters should have higher CV
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2) * 0.1
        cluster2 = np.random.randn(50, 2) * 0.1 + 10
        descriptors = np.vstack([cluster1, cluster2])
        
        cv = archive_uniformity(descriptors, k=4)
        # Clustered data has non-uniform spacing
        # Points within clusters are close, but between clusters are far
        assert cv > 0.1
    
    def test_small_archive(self):
        # k=5 with only 5 points
        descriptors = np.random.randn(5, 2)
        result = archive_uniformity(descriptors, k=5)
        assert result == 0.0  # Not enough points


class TestCoverage:
    """Tests for grid-based coverage calculation."""
    
    def test_full_coverage_2d(self):
        # Points covering all cells in a 2x2 grid
        descriptors = np.array([
            [0.1, 0.1],  # Cell (0,0)
            [0.1, 0.6],  # Cell (0,1)
            [0.6, 0.1],  # Cell (1,0)
            [0.6, 0.6],  # Cell (1,1)
        ])
        bounds_min = np.array([0.0, 0.0])
        bounds_max = np.array([1.0, 1.0])
        
        coverage = archive_coverage(descriptors, bounds_min, bounds_max, grid_resolution=2)
        assert coverage == 1.0  # All 4 cells occupied
    
    def test_partial_coverage(self):
        # Only 2 cells in 2x2 grid
        descriptors = np.array([
            [0.1, 0.1],
            [0.6, 0.6],
        ])
        bounds_min = np.array([0.0, 0.0])
        bounds_max = np.array([1.0, 1.0])
        
        coverage = archive_coverage(descriptors, bounds_min, bounds_max, grid_resolution=2)
        assert coverage == 0.5  # 2 of 4 cells
    
    def test_high_dim_coverage(self):
        # For D>6, should return raw count
        np.random.seed(42)
        descriptors = np.random.randn(1000, 10)
        bounds_min = np.min(descriptors, axis=0)
        bounds_max = np.max(descriptors, axis=0)
        
        result = archive_coverage(descriptors, bounds_min, bounds_max, grid_resolution=5)
        # Returns count of unique cells, not ratio
        assert result > 0 and result <= 1000


class TestEpsilonCoverage:
    """Tests for epsilon-coverage in high-dimensional spaces."""
    
    def test_dense_archive(self):
        # Dense random points should cover most reference samples
        np.random.seed(42)
        descriptors = np.random.rand(1000, 5)  # Unit hypercube
        
        coverage = epsilon_coverage(
            descriptors,
            epsilon=0.5,  # Large epsilon
            reference_samples=1000,
            bounds_min=np.zeros(5),
            bounds_max=np.ones(5),
            seed=42
        )
        
        # With large epsilon and 1000 points in [0,1]^5, should cover most
        assert coverage > 0.8
    
    def test_sparse_archive(self):
        # Very sparse archive should cover little
        np.random.seed(42)
        descriptors = np.random.rand(10, 5)
        
        coverage = epsilon_coverage(
            descriptors,
            epsilon=0.01,  # Very small epsilon
            reference_samples=1000,
            bounds_min=np.zeros(5),
            bounds_max=np.ones(5),
            seed=42
        )
        
        # Sparse + small epsilon = low coverage
        assert coverage < 0.1


class TestSumKNNDistances:
    """Tests for sum of k-NN distances metric."""
    
    def test_two_points(self):
        descriptors = np.array([[0.0], [1.0]])
        result = sum_of_knn_distances(descriptors, k=1)
        # Each point's nearest neighbor is 1 unit away, sum = 2
        assert result == 2.0
    
    def test_clustered_vs_spread(self):
        # Spread out points should have larger sum
        clustered = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        spread = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        
        sum_clustered = sum_of_knn_distances(clustered, k=1)
        sum_spread = sum_of_knn_distances(spread, k=1)
        
        assert sum_spread > sum_clustered


class TestComputeAllMetrics:
    """Tests for the aggregate compute_all_metrics function."""
    
    def test_returns_all_expected_keys(self):
        np.random.seed(42)
        fitness = np.random.rand(100)
        descriptors = np.random.rand(100, 5)
        
        metrics = compute_all_metrics(fitness, descriptors)
        
        expected_keys = {
            "qd_score", "max_fitness", "mean_fitness", "archive_size",
            "coverage", "mean_pairwise_distance", "uniformity_cv"
        }
        
        assert set(metrics.keys()) == expected_keys
    
    def test_consistent_with_individual_functions(self):
        np.random.seed(42)
        fitness = np.random.rand(100)
        descriptors = np.random.rand(100, 5)
        
        metrics = compute_all_metrics(fitness, descriptors)
        
        assert metrics["qd_score"] == qd_score(fitness)
        assert metrics["max_fitness"] == max_fitness(fitness)
        assert metrics["mean_fitness"] == mean_fitness(fitness)
        assert metrics["archive_size"] == len(fitness)


class TestAggregateRuns:
    """Tests for multi-run aggregation."""
    
    def test_mean_and_std_calculation(self):
        runs = [
            {"qd_score": 10.0, "max_fitness": 1.0},
            {"qd_score": 20.0, "max_fitness": 2.0},
            {"qd_score": 30.0, "max_fitness": 3.0},
        ]
        
        aggregated = aggregate_runs(runs)
        
        # Mean of [10, 20, 30] = 20
        assert aggregated["qd_score"][0] == 20.0
        # Std of [10, 20, 30] ≈ 8.165
        assert np.isclose(aggregated["qd_score"][1], np.std([10, 20, 30]))
    
    def test_empty_list(self):
        result = aggregate_runs([])
        assert result == {}


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_element_archive(self):
        fitness = np.array([1.0])
        descriptors = np.array([[0.5, 0.5]])
        
        metrics = compute_all_metrics(fitness, descriptors)
        
        assert metrics["archive_size"] == 1
        assert metrics["mean_pairwise_distance"] == 0.0
    
    def test_identical_descriptors(self):
        # All descriptors are the same
        descriptors = np.ones((100, 5))
        fitness = np.random.rand(100)
        
        metrics = compute_all_metrics(fitness, descriptors)
        
        # All pairwise distances are 0
        assert metrics["mean_pairwise_distance"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
