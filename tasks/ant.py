import gymnasium as gym
import numpy as np
from policy import TanhMLP
from concurrent.futures import ProcessPoolExecutor

# --- GLOBAL WORKER STATE ---
_env = None

def worker_init():
    """
    Runs once per worker process to initialize the persistent environment.
    """
    global _env
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
    
    # 1. Deterministic Reset
    # We pass the seed from the main process to ensure reproducibility
    obs, _ = _env.reset(seed=int(seed))
    
    input_dim = _env.observation_space.shape[0]
    output_dim = _env.action_space.shape[0]
    policy = TanhMLP(input_dim, output_dim, hidden_dim=64)
    
    total_reward = 0
    done = False
    trunc = False
    steps = 0
    
    # Restore to 1000 steps for fair comparison (or 250 for fast debug)
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

class AntTask:
    def __init__(self, workers=30):
        # 1. Dummy env for shapes
        temp_env = gym.make("Ant-v4")
        self.input_dim = temp_env.observation_space.shape[0]
        self.output_dim = temp_env.action_space.shape[0]
        temp_env.close()

        self.policy = TanhMLP(self.input_dim, self.output_dim, hidden_dim=64)
        self.param_count = self.policy.total_weights
        self.genome_size = self.param_count
        self.workers = workers
        
        # 2. START THE POOL ONCE (Crucial for Speed)
        print(f"⚙️ Spinning up {workers} persistent worker processes...")
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=worker_init
        )

    def evaluate_batch(self, genomes, current_gen_seed):
        """
        evaluates a batch of genomes using deterministic seeds.
        current_gen_seed: An integer seed unique to this generation.
        """
        batch_size = len(genomes)
        
        # Generate stable seeds for this specific batch based on the gen seed
        rng = np.random.default_rng(current_gen_seed)
        seeds = rng.integers(0, 1_000_000_000, size=batch_size)
        
        # Zip genomes with their assigned seeds
        jobs = zip(genomes, seeds)
        
        # Execute
        results = list(self.executor.map(eval_one_ant_job, jobs))

        fitnesses = np.array([r[0] for r in results])
        descriptors = np.array([r[1] for r in results])
        return fitnesses, descriptors

    def close(self):
        print("🔌 Shutting down worker pool...")
        self.executor.shutdown()
