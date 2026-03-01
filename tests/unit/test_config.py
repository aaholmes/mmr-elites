"""Tests for config module."""

import pytest

from mmr_elites.utils.config import ExperimentConfig


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig()
        assert config.task == "arm"
        assert config.n_dof == 20
        assert config.algorithm == "mmr_elites"
        assert config.archive_size == 1000
        assert config.lambda_val == 0.5
        assert config.generations == 1000
        assert config.batch_size == 200
        assert config.mutation_sigma == 0.1
        assert config.seed == 42
        assert config.log_interval == 100

    def test_post_init_syncs_niches(self):
        config = ExperimentConfig(archive_size=500, n_niches=999)
        assert config.n_niches == 500

    def test_post_init_matching(self):
        config = ExperimentConfig(archive_size=500, n_niches=500)
        assert config.n_niches == 500

    def test_from_dict(self):
        d = {"task": "rastrigin", "n_dof": 10, "generations": 500}
        config = ExperimentConfig.from_dict(d)
        assert config.task == "rastrigin"
        assert config.n_dof == 10
        assert config.generations == 500

    def test_from_dict_ignores_unknown(self):
        d = {"task": "arm", "unknown_key": "value", "another_bad": 123}
        config = ExperimentConfig.from_dict(d)
        assert config.task == "arm"
        assert not hasattr(config, "unknown_key")

    def test_to_dict(self):
        config = ExperimentConfig(task="rastrigin", n_dof=5)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["task"] == "rastrigin"
        assert d["n_dof"] == 5
        assert "archive_size" in d

    def test_roundtrip(self):
        original = ExperimentConfig(
            task="rastrigin", n_dof=10, lambda_val=0.7, generations=500
        )
        d = original.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.task == original.task
        assert restored.n_dof == original.n_dof
        assert restored.lambda_val == original.lambda_val
        assert restored.generations == original.generations
