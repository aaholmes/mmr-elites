"""
Integration tests for full experiment pipeline.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.mark.slow
class TestExperimentPipeline:
    """Tests for complete experiment runs."""
    
    def test_dimensionality_scaling_runs(self):
        """Dimensionality scaling experiment should complete."""
        from experiments.dimensionality_scaling import run_dimensionality_scaling
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_dimensionality_scaling(
                dimensions=[5, 10],
                n_seeds=1,
                generations=10,
                output_dir=Path(tmpdir),
            )
            
            assert 5 in results
            assert 10 in results
            assert "MMR-Elites" in results[5]
    
    def test_lambda_ablation_runs(self):
        """Lambda ablation experiment should complete."""
        from experiments.lambda_ablation import run_lambda_ablation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_lambda_ablation(
                lambda_values=[0.0, 0.5, 1.0],
                n_seeds=1,
                generations=10,
                output_dir=Path(tmpdir),
            )
            
            assert 0.0 in results
            assert 0.5 in results
            assert 1.0 in results


class TestResultsFormat:
    """Tests for result format consistency."""
    
    def test_all_algorithms_same_format(self):
        """All algorithms should return same result format."""
        from mmr_elites.tasks.arm import ArmTask
        from mmr_elites.algorithms import (
            run_mmr_elites, run_map_elites, run_cvt_map_elites, run_random_search
        )
        
        task = ArmTask(n_dof=5)
        
        results = [
            run_mmr_elites(task, generations=10, seed=42),
            run_map_elites(task, generations=10, seed=42),
            run_cvt_map_elites(task, generations=10, seed=42),
            run_random_search(task, generations=10, seed=42),
        ]
        
        required_keys = {"algorithm", "seed", "runtime", "final_metrics", "history"}
        required_metrics = {"qd_score", "mean_fitness", "max_fitness"}
        
        for r in results:
            assert required_keys.issubset(set(r.keys()))
            assert required_metrics.issubset(set(r["final_metrics"].keys()))
