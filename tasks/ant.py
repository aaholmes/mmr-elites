import gymnasium as gym
import numpy as np
from policy import TanhMLP

def eval_one_ant(genome):
    # CHANGE 1: terminate_when_unhealthy=False
    # This lets the ant wiggle on the floor instead of dying immediately.
    env = gym.make("Ant-v4", terminate_when_unhealthy=False)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    policy = TanhMLP(input_dim, output_dim, hidden_dim=64)
    
    obs, _ = env.reset()
    total_reward = 0
    done = False
    trunc = False
    
    steps = 0
    MAX_STEPS = 1000 
    
    while not (done or trunc) and steps < MAX_STEPS:
        action = policy.forward(obs, genome)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated
        trunc = truncated
        steps += 1
    
    # Robust Descriptor Access
    x_pos = env.unwrapped.data.qpos[0]
    y_pos = env.unwrapped.data.qpos[1]
    
    env.close()
    
    return total_reward, np.array([x_pos, y_pos])

class AntTask:
    def __init__(self):
        env = gym.make("Ant-v4")
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.shape[0]
        env.close()
        
        self.policy = TanhMLP(self.input_dim, self.output_dim, hidden_dim=64)
        self.param_count = self.policy.total_weights
