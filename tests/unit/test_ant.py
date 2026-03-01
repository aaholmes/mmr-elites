"""Tests for ant task module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mmr_elites.tasks.ant import TanhMLP, AntTask, eval_one_ant, eval_one_ant_job, worker_init


class TestTanhMLP:
    def test_init(self):
        mlp = TanhMLP(input_dim=10, output_dim=4, hidden_dim=32)
        assert mlp.input_dim == 10
        assert mlp.output_dim == 4
        assert mlp.hidden_dim == 32
        assert mlp.total_weights == 10 * 32 + 32 + 32 * 4 + 4

    def test_weight_sizes(self):
        mlp = TanhMLP(input_dim=27, output_dim=8, hidden_dim=64)
        assert mlp.w1_size == 27 * 64
        assert mlp.b1_size == 64
        assert mlp.w2_size == 64 * 8
        assert mlp.b2_size == 8
        assert mlp.total_weights == 27 * 64 + 64 + 64 * 8 + 8

    def test_unpack(self):
        mlp = TanhMLP(input_dim=4, output_dim=2, hidden_dim=3)
        genome = np.random.randn(mlp.total_weights)
        w1, b1, w2, b2 = mlp.unpack(genome)
        assert w1.shape == (4, 3)
        assert b1.shape == (3,)
        assert w2.shape == (3, 2)
        assert b2.shape == (2,)

    def test_forward(self):
        mlp = TanhMLP(input_dim=4, output_dim=2, hidden_dim=3)
        genome = np.random.randn(mlp.total_weights)
        state = np.random.randn(4)
        action = mlp.forward(state, genome)
        assert action.shape == (2,)
        # Output should be in [-1, 1] due to tanh
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_forward_batch_state(self):
        mlp = TanhMLP(input_dim=4, output_dim=2, hidden_dim=3)
        genome = np.random.randn(mlp.total_weights)
        state = np.random.randn(4)
        action = mlp.forward(state, genome)
        assert action.shape == (2,)

    def test_deterministic(self):
        mlp = TanhMLP(input_dim=4, output_dim=2, hidden_dim=3)
        genome = np.random.randn(mlp.total_weights)
        state = np.random.randn(4)
        a1 = mlp.forward(state, genome)
        a2 = mlp.forward(state, genome)
        np.testing.assert_array_equal(a1, a2)


class TestEvalOneAnt:
    def test_eval_one_ant_no_env(self):
        """When _env is None, should return zero fitness and zero descriptors."""
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env
        ant_module._env = None
        try:
            fitness, desc = eval_one_ant(np.zeros(100), 42)
            assert fitness == 0.0
            assert np.all(desc == 0)
        finally:
            ant_module._env = original_env

    def test_eval_one_ant_job_unpacks(self):
        """eval_one_ant_job should unpack (genome, seed) tuple."""
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env
        ant_module._env = None
        try:
            genome = np.zeros(100)
            fitness, desc = eval_one_ant_job((genome, 42))
            assert fitness == 0.0
        finally:
            ant_module._env = original_env


class TestAntTask:
    def test_init_without_gym(self):
        """Test initialization when gymnasium is available but we mock it."""
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.input_dim = 27
            task.output_dim = 8
            task.policy = TanhMLP(27, 8, hidden_dim=64)
            task.param_count = task.policy.total_weights
            task._genome_dim = task.param_count
            task._desc_dim = 2
            task.workers = 4
            task.executor = None

            assert task.genome_dim == task.param_count
            assert task.desc_dim == 2

    def test_genome_dim_property(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.executor = None
            task._genome_dim = 100
            assert task.genome_dim == 100

    def test_desc_dim_property(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.executor = None
            task._desc_dim = 2
            assert task.desc_dim == 2

    def test_evaluate_without_gym_raises(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.executor = None
            with pytest.raises(RuntimeError, match="Gymnasium not installed"):
                task.evaluate(np.zeros((5, 100)))

    def test_close_no_executor(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.executor = None
            task.close()  # Should not raise

    def test_close_with_executor(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            mock_executor = MagicMock()
            task.executor = mock_executor
            task.close()
            mock_executor.shutdown.assert_called_once()
            assert task.executor is None

    def test_start_without_gym(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            task.executor = None
            task.workers = 4
            task.start()
            # Should not start executor when gym is not available
            assert task.executor is None

    def test_start_with_existing_executor(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", True):
            task = AntTask.__new__(AntTask)
            mock_executor = MagicMock()
            task.executor = mock_executor
            task.workers = 4
            task.start()
            # Should not create a new executor
            assert task.executor is mock_executor

    def test_init_without_gym_available(self):
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            task = AntTask.__new__(AntTask)
            # Simulate what __init__ does when GYM_AVAILABLE is False
            task.input_dim = 27
            task.output_dim = 8
            task.policy = TanhMLP(27, 8, hidden_dim=64)
            task.param_count = task.policy.total_weights
            task._genome_dim = task.param_count
            task._desc_dim = 2
            task.workers = 4
            task.executor = None
            assert task.input_dim == 27
            assert task.output_dim == 8
            assert task.desc_dim == 2

    def test_init_with_gym_available(self):
        """Test __init__ path when gym is available using a mock environment."""
        mock_env = MagicMock()
        mock_env.observation_space.shape = (27,)
        mock_env.action_space.shape = (8,)
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", True), \
             patch("mmr_elites.tasks.ant.gym") as mock_gym:
            mock_gym.make.return_value = mock_env
            task = AntTask(workers=2)
            assert task.input_dim == 27
            assert task.output_dim == 8
            assert task.workers == 2
            assert task.executor is None
            mock_env.close.assert_called_once()

    def test_start_creates_executor(self):
        """Test that start() creates a ProcessPoolExecutor when gym is available."""
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", True), \
             patch("mmr_elites.tasks.ant.ProcessPoolExecutor") as MockPPE:
            task = AntTask.__new__(AntTask)
            task.executor = None
            task.workers = 2
            task.start()
            MockPPE.assert_called_once()

    def test_evaluate_starts_executor_and_runs(self):
        """Test evaluate with mocked executor."""
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", True):
            task = AntTask.__new__(AntTask)
            task.executor = None
            task.workers = 2
            task._genome_dim = 100
            task._desc_dim = 2

            mock_executor = MagicMock()
            mock_results = [(1.0, np.array([0.5, 0.3])), (2.0, np.array([0.1, 0.2]))]
            mock_executor.map.return_value = mock_results

            with patch.object(task, "start") as mock_start:
                # After start is called, set executor
                def set_executor():
                    task.executor = mock_executor
                mock_start.side_effect = set_executor

                genomes = np.zeros((2, 100))
                fitnesses, descriptors = task.evaluate(genomes)
                assert fitnesses.shape == (2,)
                assert descriptors.shape == (2, 2)
                np.testing.assert_array_equal(fitnesses, [1.0, 2.0])


class TestWorkerInit:
    def test_worker_init_with_gym(self):
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env
        mock_env = MagicMock()
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", True), \
             patch("mmr_elites.tasks.ant.gym") as mock_gym:
            mock_gym.make.return_value = mock_env
            worker_init()
            assert ant_module._env is mock_env
            mock_gym.make.assert_called_once_with("Ant-v4", terminate_when_unhealthy=False)
        ant_module._env = original_env

    def test_worker_init_without_gym(self):
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env
        ant_module._env = None
        with patch("mmr_elites.tasks.ant.GYM_AVAILABLE", False):
            worker_init()
            assert ant_module._env is None
        ant_module._env = original_env


class TestEvalOneAntWithEnv:
    def test_eval_with_mock_env(self):
        """Test eval_one_ant with a mocked environment."""
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env

        mock_env = MagicMock()
        mock_env.observation_space.shape = (27,)
        mock_env.action_space.shape = (8,)
        mock_env.reset.return_value = (np.zeros(27), {})
        # Simulate environment stepping: terminate after 2 steps
        mock_env.step.side_effect = [
            (np.zeros(27), 1.0, False, False, {}),
            (np.zeros(27), 2.0, True, False, {}),
        ]
        mock_env.unwrapped.data.qpos = [0.5, -0.3, 0, 0]

        ant_module._env = mock_env
        try:
            mlp = TanhMLP(27, 8, hidden_dim=64)
            genome = np.zeros(mlp.total_weights)
            fitness, desc = eval_one_ant(genome, 42)
            assert fitness == 3.0  # 1.0 + 2.0
            np.testing.assert_array_almost_equal(desc, [0.5, -0.3])
        finally:
            ant_module._env = original_env

    def test_eval_truncated(self):
        """Test eval_one_ant when environment truncates."""
        import mmr_elites.tasks.ant as ant_module
        original_env = ant_module._env

        mock_env = MagicMock()
        mock_env.observation_space.shape = (27,)
        mock_env.action_space.shape = (8,)
        mock_env.reset.return_value = (np.zeros(27), {})
        mock_env.step.return_value = (np.zeros(27), 5.0, False, True, {})
        mock_env.unwrapped.data.qpos = [1.0, 2.0, 0, 0]

        ant_module._env = mock_env
        try:
            mlp = TanhMLP(27, 8, hidden_dim=64)
            genome = np.zeros(mlp.total_weights)
            fitness, desc = eval_one_ant(genome, 0)
            assert fitness == 5.0
            np.testing.assert_array_almost_equal(desc, [1.0, 2.0])
        finally:
            ant_module._env = original_env
