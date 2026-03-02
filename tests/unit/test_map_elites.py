"""
Unit tests for MAP-Elites implementation.

Tests verify:
1. Grid cell assignment
2. Archive update logic (keep best per cell)
3. Parent sampling
4. Correct behavior with high-D descriptors
"""

import numpy as np
import pytest

from mmr_elites.algorithms.base import ExperimentConfig
from mmr_elites.algorithms.map_elites import MAPElites


class TestGridCellAssignment:
    """Tests for descriptor-to-cell mapping."""

    def test_basic_assignment(self):
        """Test basic cell assignment."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=10)

        # Corners
        assert me._get_cell(np.array([0.0, 0.0])) == (0, 0)
        assert me._get_cell(np.array([0.99, 0.99])) == (9, 9)

        # Middle
        assert me._get_cell(np.array([0.5, 0.5])) == (5, 5)

    def test_clipping(self):
        """Values outside [0,1] should be clipped."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=10)

        # Out of bounds
        assert me._get_cell(np.array([-0.1, 1.5])) == (0, 9)

    def test_high_dimensional(self):
        """Test with high-dimensional descriptors."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=3)

        desc = np.ones(20) * 0.5
        cell = me._get_cell(desc)

        assert len(cell) == 20
        assert all(c == 1 for c in cell)


class TestArchiveUpdate:
    """Tests for archive update logic."""

    def test_add_to_empty_cell(self):
        """Adding to empty cell should succeed."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=10)

        genome = np.array([0.1, 0.2])
        fitness = 0.5
        descriptor = np.array([0.5, 0.5])

        added = me._add_to_archive(genome, fitness, descriptor)

        assert added is True
        assert len(me.archive) == 1

    def test_replace_worse_fitness(self):
        """Better fitness should replace worse."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=10)

        # Add initial
        me._add_to_archive(np.array([1.0]), 0.3, np.array([0.5, 0.5]))

        # Add better to same cell
        added = me._add_to_archive(np.array([2.0]), 0.8, np.array([0.51, 0.51]))

        assert added is True
        cell = (5, 5)
        assert me.archive[cell][1] == 0.8  # Fitness updated

    def test_reject_worse_fitness(self):
        """Worse fitness should not replace better."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=10)

        # Add initial
        me._add_to_archive(np.array([1.0]), 0.8, np.array([0.5, 0.5]))

        # Try to add worse
        added = me._add_to_archive(np.array([2.0]), 0.3, np.array([0.51, 0.51]))

        assert added is False
        cell = (5, 5)
        assert me.archive[cell][1] == 0.8  # Unchanged


class TestParentSampling:
    """Tests for parent selection."""

    def test_uniform_sampling(self):
        """Parents should be sampled uniformly from archive."""
        config = ExperimentConfig()
        me = MAPElites(config, bins_per_dim=5)

        # Add 5 solutions to different cells
        for i in range(5):
            desc = np.ones(2) * (i * 0.2 + 0.1)
            me._add_to_archive(np.array([float(i)]), 0.5, desc)

        np.random.seed(42)
        parents = me._sample_parents(1000)

        # Check roughly uniform (each cell ~200 samples)
        counts = {}
        for p in parents:
            key = tuple(p)
            counts[key] = counts.get(key, 0) + 1

        for count in counts.values():
            assert 100 < count < 300  # Roughly uniform
        assert len(counts) == 5
