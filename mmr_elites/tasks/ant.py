"""
Ant Locomotion Task (MuJoCo).
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np

from .base import Task

try:
    import gymnasium as gym

    GYM_AVAILABLE = True
except ImportError:
    gym = None
    GYM_AVAILABLE = False


class TanhMLP:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Calculate shapes
        self.w1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim * output_dim
        self.b2_size = output_dim

        self.total_weights = self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def unpack(self, genome):
        """Slice the flat genome vector into weight matrices."""
        idx = 0
        w1 = genome[idx : idx + self.w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += self.w1_size

        b1 = genome[idx : idx + self.b1_size]
        idx += self.b1_size

        w2 = genome[idx : idx + self.w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += self.w2_size

        b2 = genome[idx : idx + self.b2_size]

        return w1, b1, w2, b2

    def forward(self, state, genome):
        w1, b1, w2, b2 = self.unpack(genome)

        # Layer 1
        x = np.tanh(state @ w1 + b1)
        # Layer 2 (Output)
        action = np.tanh(x @ w2 + b2)

        return action


# --- GLOBAL WORKER STATE ---
_env = None


def worker_init():
    """
    Runs once per worker process to initialize the persistent environment.
    """
    global _env
    if GYM_AVAILABLE:
        # terminate_when_unhealthy=False is crucial for Ant-v4 exploration
        _env = gym.make("Ant-v4", terminate_when_unhealthy=False)


def eval_one_ant_job(args):
    """
    Helper to unpack arguments since map() only passes one item.
    args: (genome, seed)
    """
    genome, seed = args
    return eval_one_ant(genome, seed)


def eval_one_ant(genome, seed):
    global _env

    if _env is None:
        return 0.0, np.zeros(2)

    # 1. Deterministic Reset
    obs, _ = _env.reset(seed=int(seed))

    input_dim = _env.observation_space.shape[0]
    output_dim = _env.action_space.shape[0]
    policy = TanhMLP(input_dim, output_dim, hidden_dim=64)

    total_reward = 0
    done = False
    trunc = False
    steps = 0

    MAX_STEPS = 1000

    while not (done or trunc) and steps < MAX_STEPS:
        action = policy.forward(obs, genome)
        obs, reward, terminated, truncated, _ = _env.step(action)
        total_reward += reward
        done = terminated
        trunc = truncated
        steps += 1

    x_pos = _env.unwrapped.data.qpos[0]
    y_pos = _env.unwrapped.data.qpos[1]

    return total_reward, np.array([x_pos, y_pos])


class AntTask(Task):
    def __init__(self, workers=4):
        if not GYM_AVAILABLE:
            print("Warning: Gymnasium not installed. AntTask will fail if used.")
            self.input_dim = 27
            self.output_dim = 8
        else:
            # 1. Dummy env for shapes
            temp_env = gym.make("Ant-v4")
            self.input_dim = temp_env.observation_space.shape[0]
            self.output_dim = temp_env.action_space.shape[0]
            temp_env.close()

        self.policy = TanhMLP(self.input_dim, self.output_dim, hidden_dim=64)
        self.param_count = self.policy.total_weights
        self._genome_dim = self.param_count
        self._desc_dim = 2  # x, y

        self.workers = workers
        self.executor = None

    @property
    def genome_dim(self) -> int:
        return self._genome_dim

    @property
    def desc_dim(self) -> int:
        return self._desc_dim

    def start(self):
        """Explicitly start the worker pool."""
        if self.executor is None and GYM_AVAILABLE:
            print(f"⚙️ Spinning up {self.workers} persistent worker processes...")
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers, initializer=worker_init
            )

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of genomes.
        """
        if not GYM_AVAILABLE:
            raise RuntimeError("Gymnasium not installed.")

        if self.executor is None:
            self.start()

        batch_size = len(genomes)

        # Generate stable seeds for this batch
        # We need a seed for reproducibility.
        # Ideally, the seed should be passed in, but Task interface is simple.
        # We'll use a random seed here, but in the loop, numpy is seeded.
        rng = np.random.default_rng()
        # NOTE: This breaks exact reproducibility if called from seeded loop without passing state?
        # Actually, if np.random is seeded in the main loop, rng here will be different?
        # No, default_rng() creates a NEW generator.
        # We should use np.random to generate seeds if we want main seed to control this.
        seeds = np.random.randint(0, 1_000_000_000, size=batch_size)

        # Zip genomes with their assigned seeds
        jobs = zip(genomes, seeds)

        # Execute
        results = list(self.executor.map(eval_one_ant_job, jobs))

        fitnesses = np.array([r[0] for r in results])
        descriptors = np.array([r[1] for r in results])

        # Normalize descriptors?
        # Ant moves in infinite space. Typically we don't normalize or we use a large range.
        # For uniformity metrics, we might want normalization, but raw x/y is fine for coverage if bounds provided.

        return fitnesses, descriptors

    def close(self):
        if self.executor:
            print("🔌 Shutting down worker pool...")
            self.executor.shutdown()
            self.executor = None

    def __del__(self):
        self.close()
