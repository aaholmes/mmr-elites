"""
Unit tests for benchmark tasks.

Tests verify:
1. Fitness computation
2. Descriptor generation
3. Collision detection (for arm task)
4. Output shapes and ranges
"""

import pytest
import numpy as np


class TestArmTask:
    """Tests for N-DOF arm task."""
    
    def test_output_shapes(self, arm_task):
        """Verify output shapes."""
        genomes = np.random.uniform(-np.pi, np.pi, (100, 5))
        fitness, descriptors = arm_task.evaluate(genomes)
        
        assert fitness.shape == (100,)
        assert descriptors.shape == (100, 5)  # High-D descriptor
    
    def test_fitness_range(self, arm_task):
        """Fitness should be in [0, 1]."""
        genomes = np.random.uniform(-np.pi, np.pi, (1000, 5))
        fitness, _ = arm_task.evaluate(genomes)
        
        assert np.all(fitness >= 0)
        assert np.all(fitness <= 1)
    
    def test_descriptor_range(self, arm_task):
        """Descriptors should be in [0, 1] (normalized)."""
        genomes = np.random.uniform(-np.pi, np.pi, (1000, 5))
        _, descriptors = arm_task.evaluate(genomes)
        
        assert np.all(descriptors >= 0)
        assert np.all(descriptors <= 1)
    
    def test_collision_reduces_fitness(self, arm_task):
        """Colliding configurations should have fitness = 0."""
        # This requires knowing the obstacle position
        # Generate many random configs and check collision rate
        genomes = np.random.uniform(-np.pi, np.pi, (10000, 5))
        fitness, _ = arm_task.evaluate(genomes)
        
        # Should have some zeros (collisions)
        collision_rate = np.mean(fitness == 0)
        assert collision_rate > 0  # At least some collisions
        assert collision_rate < 1  # Not all collisions
    
    def test_optimal_fitness_achievable(self, arm_task):
        """Should be possible to achieve high fitness."""
        # Run optimization to find good solution
        best_fitness = 0
        for _ in range(100):
            genomes = np.random.uniform(-np.pi, np.pi, (1000, 5))
            fitness, _ = arm_task.evaluate(genomes)
            best_fitness = max(best_fitness, np.max(fitness))
        
        assert best_fitness > 0.5  # Should achieve reasonable fitness


class TestRastriginTask:
    """Tests for Rastrigin benchmark."""
    
    def test_output_shapes(self):
        """Verify output shapes."""
        from mmr_elites.tasks.rastrigin import RastriginTask
        
        task = RastriginTask(n_dim=10)
        genomes = np.random.uniform(-np.pi, np.pi, (100, 10))
        fitness, descriptors = task.evaluate(genomes)
        
        assert fitness.shape == (100,)
        assert descriptors.shape == (100, 10)
    
    def test_global_optimum(self):
        """Fitness at origin should be highest."""
        from mmr_elites.tasks.rastrigin import RastriginTask
        
        task = RastriginTask(n_dim=5)
        
        # Optimal: all zeros (mapped from genome space)
        optimal = np.zeros((1, 5))
        random = np.random.uniform(-np.pi, np.pi, (100, 5))
        
        opt_fitness, _ = task.evaluate(optimal)
        rand_fitness, _ = task.evaluate(random)
        
        assert opt_fitness[0] >= np.max(rand_fitness) - 0.01
