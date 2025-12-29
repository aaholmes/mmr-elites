"""
20-DOF Arm with HIGH-DIMENSIONAL behavior descriptor.

This is the CRITICAL task for demonstrating MUSE-QD's advantage over MAP-Elites.
The behavior descriptor is the 20 joint angles themselves (20D space).

In this setup:
- MAP-Elites with even 3 bins/dim = 3^20 ≈ 3.5 billion cells → degrades to random search
- MUSE-QD maintains exactly K diverse solutions regardless of dimensionality

This is where MUSE-QD WINS.
"""

import numpy as np
from typing import Tuple


class Arm20HighDimTask:
    """
    20-DOF Planar Arm where behavior descriptor = normalized joint angles (20D).
    
    Fitness: Distance of end-effector to target (with collision penalty)
    Descriptor: All 20 joint angles, normalized to [0, 1]
    
    This task demonstrates the curse of dimensionality in behavior spaces.
    """
    
    def __init__(self, target_pos: Tuple[float, float] = (0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof  # Total arm length = 1.0
        self.target_pos = np.array(target_pos)
        
        # Obstacle (wall blocking direct path to target)
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]
        
        # Behavior space bounds (normalized joint angles)
        self.desc_bounds_min = np.zeros(20)
        self.desc_bounds_max = np.ones(20)
    
    def forward_kinematics_batch(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute positions of all joints for a batch of configurations.
        
        Args:
            joints: Joint angles (batch_size, 20) in radians
            
        Returns:
            Joint coordinates (batch_size, 20, 2) - x,y for each joint
        """
        # Cumulative angles (each joint angle is relative to previous)
        angles = np.cumsum(joints, axis=1)
        
        # Link endpoint offsets
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        
        # Cumulative positions (from base at origin)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        
        return np.stack([x, y], axis=2)
    
    def check_collisions_batch(self, joint_coords: np.ndarray) -> np.ndarray:
        """
        Check which configurations collide with the obstacle.
        
        Args:
            joint_coords: Joint positions (batch_size, 20, 2)
            
        Returns:
            Boolean array (batch_size,) - True if collision
        """
        batch_size = joint_coords.shape[0]
        
        # Add origin point to create full path
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)  # (batch, 21, 2)
        
        # Test 1: Check if any joint is inside the obstacle box
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_inside = np.any(p_in_x & p_in_y, axis=1)
        
        # Test 2: Check line segment intersections with box edges
        A = points[:, :-1, :]  # Start points (batch, 20, 2)
        B = points[:, 1:, :]   # End points (batch, 20, 2)
        
        Ax, Ay = A[:, :, 0], A[:, :, 1]
        Bx, By = B[:, :, 0], B[:, :, 1]
        dx, dy = Bx - Ax, By - Ay
        
        # Avoid division by zero
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        
        def check_vertical(wall_x, y_min, y_max):
            """Check intersection with vertical line x = wall_x."""
            t = (wall_x - Ax) / dx
            y_at_t = Ay + t * dy
            return (t >= 0) & (t <= 1) & (y_at_t >= y_min) & (y_at_t <= y_max)
        
        def check_horizontal(wall_y, x_min, x_max):
            """Check intersection with horizontal line y = wall_y."""
            t = (wall_y - Ay) / dy
            x_at_t = Ax + t * dx
            return (t >= 0) & (t <= 1) & (x_at_t >= x_min) & (x_at_t <= x_max)
        
        # Check all 4 box edges
        hit = (
            check_vertical(self.box_x[0], self.box_y[0], self.box_y[1]) |   # Left
            check_vertical(self.box_x[1], self.box_y[0], self.box_y[1]) |   # Right
            check_horizontal(self.box_y[0], self.box_x[0], self.box_x[1]) | # Bottom
            check_horizontal(self.box_y[1], self.box_x[0], self.box_x[1])   # Top
        )
        any_intersection = np.any(hit, axis=1)
        
        return any_inside | any_intersection
    
    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of genomes.
        
        Args:
            genomes: Joint angles (batch_size, 20) in radians [-π, π]
            
        Returns:
            fitness: Fitness values (batch_size,)
            descriptors: Behavior descriptors (batch_size, 20) - normalized joint angles
        """
        # Forward kinematics
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]  # End-effector positions (batch, 2)
        
        # Fitness: 1 - distance to target (higher is better)
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = 1.0 - dists
        fitness = np.maximum(fitness, 0.0)
        
        # Collision penalty
        collides = self.check_collisions_batch(joint_coords)
        fitness[collides] = 0.0
        
        # CRITICAL: Behavior descriptor = normalized joint angles (20D)
        # This is what makes MAP-Elites fail - 20D behavior space!
        descriptors = (genomes + np.pi) / (2 * np.pi)  # Normalize [-π,π] to [0,1]
        
        return fitness, descriptors
    
    def get_end_effector_positions(self, genomes: np.ndarray) -> np.ndarray:
        """Helper to get just end-effector positions for visualization."""
        joint_coords = self.forward_kinematics_batch(genomes)
        return joint_coords[:, -1, :]


# For backwards compatibility
Arm20Task = Arm20HighDimTask
