"""
Unit tests for the Rust MMR selector implementation.

Tests verify:
1. Correctness against naive O(N*K^2) implementation
2. Edge cases (K >= N, empty inputs, etc.)
3. Lambda boundary conditions (0, 0.5, 1)
4. Determinism (same input -> same output)
"""

import numpy as np
import pytest

try:
    import mmr_elites_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def naive_greedy_mmr(
    fitness: np.ndarray, descriptors: np.ndarray, target_k: int, lambda_val: float
) -> np.ndarray:
    """
    Reference implementation: O(N * K^2) naive greedy MMR.

    This is the ground truth for verifying the optimized Rust implementation.
    Normalizes fitness to [0, 1] to match the Rust backend.
    """
    n = len(fitness)
    if n <= target_k:
        return np.arange(n)

    # Normalize fitness to [0, 1]
    f_min, f_max = fitness.min(), fitness.max()
    if f_max - f_min > 1e-10:
        f_norm = (fitness - f_min) / (f_max - f_min)
    else:
        f_norm = np.full(n, 0.5)

    selected = []
    remaining = set(range(n))

    # Seed with best fitness
    best_idx = int(np.argmax(fitness))
    selected.append(best_idx)
    remaining.remove(best_idx)

    # Greedy loop
    while len(selected) < target_k and remaining:
        best_score = -np.inf
        best_candidate = None

        for idx in remaining:
            # Compute d_min to current archive
            d_min = np.min(
                [np.linalg.norm(descriptors[idx] - descriptors[s]) for s in selected]
            )

            score = (1 - lambda_val) * f_norm[idx] + lambda_val * d_min

            if score > best_score:
                best_score = score
                best_candidate = idx

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    return np.array(selected)


@pytest.mark.rust
class TestMMRSelectorCorrectness:
    """Tests that Rust implementation matches naive implementation."""

    @pytest.mark.parametrize(
        "n,k,d,lam",
        [
            (50, 10, 2, 0.5),  # Small 2D
            (100, 20, 5, 0.5),  # Medium 5D
            (200, 50, 10, 0.5),  # Larger 10D
            (100, 10, 20, 0.5),  # High-D descriptors
            (100, 10, 5, 0.0),  # Lambda = 0 (pure fitness)
            (100, 10, 5, 1.0),  # Lambda = 1 (pure diversity)
            (100, 10, 5, 0.3),  # Lambda = 0.3
            (100, 10, 5, 0.7),  # Lambda = 0.7
        ],
    )
    def test_matches_naive_implementation(self, n, k, d, lam):
        """Rust result must exactly match naive Python result."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        np.random.seed(42)
        fitness = np.random.rand(n).astype(np.float64)
        descriptors = np.random.rand(n, d).astype(np.float64)

        # Naive Python
        naive_result = naive_greedy_mmr(fitness, descriptors, k, lam)

        # Optimized Rust
        selector = mmr_elites_rs.MMRSelector(k, lam)
        rust_result = selector.select(fitness, descriptors)

        np.testing.assert_array_equal(
            naive_result,
            rust_result,
            err_msg=f"Mismatch for n={n}, k={k}, d={d}, λ={lam}",
        )

    def test_deterministic(self, small_population):
        """Same input should always produce same output."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness, descriptors = small_population
        selector = mmr_elites_rs.MMRSelector(20, 0.5)

        result1 = selector.select(fitness, descriptors)
        result2 = selector.select(fitness, descriptors)

        np.testing.assert_array_equal(result1, result2)

    def test_first_selected_is_best_fitness(self, small_population):
        """First selected element must be the one with highest fitness."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness, descriptors = small_population
        selector = mmr_elites_rs.MMRSelector(10, 0.5)

        result = selector.select(fitness, descriptors)

        assert result[0] == np.argmax(fitness)


@pytest.mark.rust
class TestMMRSelectorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_k_greater_than_n(self):
        """When K >= N, should return all indices."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        n, k = 10, 20
        fitness = np.random.rand(n).astype(np.float64)
        descriptors = np.random.rand(n, 5).astype(np.float64)

        selector = mmr_elites_rs.MMRSelector(k, 0.5)
        result = selector.select(fitness, descriptors)

        assert len(result) == n
        assert set(result) == set(range(n))

    def test_k_equals_n(self):
        """When K == N, should return all indices."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        n = 50
        fitness = np.random.rand(n).astype(np.float64)
        descriptors = np.random.rand(n, 5).astype(np.float64)

        selector = mmr_elites_rs.MMRSelector(n, 0.5)
        result = selector.select(fitness, descriptors)

        assert len(result) == n

    def test_k_equals_1(self):
        """When K == 1, should return only best fitness."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness = np.array([0.1, 0.9, 0.5]).astype(np.float64)
        descriptors = np.random.rand(3, 2).astype(np.float64)

        selector = mmr_elites_rs.MMRSelector(1, 0.5)
        result = selector.select(fitness, descriptors)

        assert len(result) == 1
        assert result[0] == 1  # Index of max fitness

    def test_identical_fitness(self):
        """When all fitness values are identical, selection by diversity."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        n = 10
        fitness = np.ones(n).astype(np.float64)
        descriptors = np.random.rand(n, 5).astype(np.float64)

        selector = mmr_elites_rs.MMRSelector(5, 0.5)
        result = selector.select(fitness, descriptors)

        # Should complete without error and return K distinct indices
        assert len(result) == 5
        assert len(set(result)) == 5

    def test_identical_descriptors(self):
        """When all descriptors are identical, selection by fitness."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        n = 10
        fitness = np.random.rand(n).astype(np.float64)
        descriptors = np.ones((n, 5)).astype(np.float64)  # All same

        selector = mmr_elites_rs.MMRSelector(5, 0.5)
        result = selector.select(fitness, descriptors)

        # d_min is always 0, so selection is by fitness only
        assert len(result) == 5
        # Top 5 by fitness should be selected
        expected = np.argsort(fitness)[-5:][::-1]
        np.testing.assert_array_equal(result, expected)


@pytest.mark.rust
class TestMMRSelectorNormalization:
    """Tests for fitness normalization in the Rust selector."""

    def test_fitness_normalization(self):
        """Selection should be invariant to fitness scale/shift.

        Passing fitness [100, 200, 300] should produce the same selection
        as [0.0, 0.5, 1.0] (with proportional descriptors), because both
        normalize to the same relative values.
        """
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        np.random.seed(42)
        n, k = 50, 10
        descriptors = np.random.rand(n, 5).astype(np.float64)

        # Fitness in [0, 1]
        fitness_unit = np.random.rand(n).astype(np.float64)

        # Same fitness scaled to [100, 200]
        fitness_scaled = fitness_unit * 100 + 100

        selector = mmr_elites_rs.MMRSelector(k, 0.5)
        result_unit = selector.select(fitness_unit, descriptors)
        result_scaled = selector.select(fitness_scaled, descriptors)

        np.testing.assert_array_equal(
            result_unit,
            result_scaled,
            err_msg="Selection should be invariant to linear fitness scaling",
        )

    def test_negative_fitness(self):
        """Selector should handle negative fitness values without crashing."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        n, k = 20, 5
        fitness = np.array([-10, -5, 0, 5, 10] * 4, dtype=np.float64)
        descriptors = np.random.rand(n, 3).astype(np.float64)

        selector = mmr_elites_rs.MMRSelector(k, 0.5)
        result = selector.select(fitness, descriptors)

        assert len(result) == k
        assert len(set(result)) == k  # All distinct
        assert result[0] == np.argmax(fitness)  # Best fitness first


@pytest.mark.rust
class TestMMRSelectorLambda:
    """Tests for lambda parameter behavior."""

    def test_lambda_zero_is_pure_fitness(self, small_population):
        """λ=0 should select top-K by fitness only."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness, descriptors = small_population
        k = 20

        selector = mmr_elites_rs.MMRSelector(k, 0.0)
        result = selector.select(fitness, descriptors)

        # Should be top-K indices by fitness
        expected = np.argsort(fitness)[-k:][::-1]
        np.testing.assert_array_equal(result, expected)

    def test_lambda_one_maximizes_diversity(self, small_population):
        """λ=1 should prioritize diversity over fitness."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness, descriptors = small_population

        # λ=0: pure fitness
        selector_fitness = mmr_elites_rs.MMRSelector(20, 0.0)
        result_fitness = selector_fitness.select(fitness, descriptors)

        # λ=1: pure diversity
        selector_diverse = mmr_elites_rs.MMRSelector(20, 1.0)
        result_diverse = selector_diverse.select(fitness, descriptors)

        # Compute mean pairwise distances
        def mean_pairwise_dist(indices):
            descs = descriptors[indices]
            from scipy.spatial.distance import cdist

            dists = cdist(descs, descs)
            return np.mean(dists[np.triu_indices(len(indices), k=1)])

        mpd_fitness = mean_pairwise_dist(result_fitness)
        mpd_diverse = mean_pairwise_dist(result_diverse)

        # λ=1 should achieve higher diversity
        assert mpd_diverse > mpd_fitness

    def test_lambda_interpolates(self, small_population):
        """Higher λ should increase diversity monotonically."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust backend not available")

        fitness, descriptors = small_population

        from scipy.spatial.distance import cdist

        def mean_pairwise_dist(indices):
            descs = descriptors[indices]
            dists = cdist(descs, descs)
            return np.mean(dists[np.triu_indices(len(indices), k=1)])

        lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
        diversities = []

        for lam in lambdas:
            selector = mmr_elites_rs.MMRSelector(20, lam)
            result = selector.select(fitness, descriptors)
            diversities.append(mean_pairwise_dist(result))

        # Check monotonic increase (with some tolerance for noise)
        for i in range(len(diversities) - 1):
            assert (
                diversities[i + 1] >= diversities[i] * 0.95
            ), f"Diversity decreased: λ={lambdas[i]}→{lambdas[i+1]}"
