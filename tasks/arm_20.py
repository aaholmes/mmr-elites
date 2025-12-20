import numpy as np

class Arm20Task:
    def __init__(self, target_pos=(0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof
        self.target_pos = np.array(target_pos)
        
        # WALL FIXED: Reduced height to [-0.25, 0.25]
        # Previous height [-0.5, 0.5] made the path length 1.29 (Impossible)
        self.wall_x_min = 0.5
        self.wall_x_max = 0.55
        self.wall_y_min = -0.25
        self.wall_y_max = 0.25

    def forward_kinematics_batch(self, joints):
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)

    def evaluate(self, genomes):
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]
        
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = 1.0 - dists
        fitness = np.maximum(fitness, 0.0)
        
        in_x = (joint_coords[:, :, 0] > self.wall_x_min) & (joint_coords[:, :, 0] < self.wall_x_max)
        in_y = (joint_coords[:, :, 1] > self.wall_y_min) & (joint_coords[:, :, 1] < self.wall_y_max)
        collides = np.any(in_x & in_y, axis=1)
        
        fitness[collides] = 0.0
        
        return fitness, tips

    def get_descriptor(self, genomes):
        _, tips = self.evaluate(genomes)
        return tips 
