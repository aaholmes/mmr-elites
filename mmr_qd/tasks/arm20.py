import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from ..task import Task

class Arm20DOF(Task):
    def __init__(self, target_pos: Tuple[float, float] = (10.0, 10.0)):
        self.num_joints = 20
        self.link_lengths = np.ones(self.num_joints) # Length 1.0 each
        self.target_pos = np.array(target_pos)
        
        # Bound genes to [-pi, pi]
        self.min_gene = -np.pi
        self.max_gene = np.pi

    def forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """
        Calculates the (x,y) position of the End Effector.
        Input: (Batch, 20) angles
        Output: (Batch, 2) coordinates
        """
        # Cumulative sum of angles to get absolute orientation of each link
        # Assuming theta_i is relative to link_{i-1}
        absolute_angles = np.cumsum(thetas, axis=1)
        
        # x = sum(L * cos(abs_angle))
        xs = self.link_lengths * np.cos(absolute_angles)
        ys = self.link_lengths * np.sin(absolute_angles)
        
        # Sum up the vectors to get the tip position
        tip_x = np.sum(xs, axis=1)
        tip_y = np.sum(ys, axis=1)
        
        return np.stack([tip_x, tip_y], axis=1)

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute Tip Positions
        tips = self.forward_kinematics(genomes)
        
        # 2. Compute Euclidean Distance to Target
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        
        # 3. Fitness: Maximize (1 - Distance), or negative distance
        # We want fitness > 0 for valid solutions generally, but for ranking 
        # negative distance is fine. Let's use negative distance.
        fitnesses = -dists 
        
        # 4. Descriptor: The behavior is the posture itself (the angles)
        # We normalize this to ensure Euclidean distance in selector makes sense.
        # Since angles are [-pi, pi], they are already roughly scaled.
        descriptors = genomes.copy()
        
        return fitnesses, descriptors

    def visualize(self, genome: np.ndarray, ax=None):
        """Helper to draw the arm for debugging/paper figures"""
        if ax is None:
            _, ax = plt.subplots()
            
        absolute_angles = np.cumsum(genome)
        xs = np.concatenate(([0], self.link_lengths * np.cos(absolute_angles)))
        ys = np.concatenate(([0], self.link_lengths * np.sin(absolute_angles)))
        
        # Cumulative path
        path_x = np.cumsum(xs)
        path_y = np.cumsum(ys)
        
        ax.plot(path_x, path_y, 'o-', linewidth=2, markersize=5)
        ax.plot(self.target_pos[0], self.target_pos[1], 'r*', markersize=15, label='Target')
        ax.set_aspect('equal')
