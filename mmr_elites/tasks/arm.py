"""
N-DOF Planar Arm Task.
"""

from typing import Optional, Tuple

import numpy as np

from .base import Task


class ArmTask(Task):
    """
    N-DOF planar arm reaching task.

    Genome: Joint angles in [-π, π]
    Fitness: Distance to target (higher is better)
    Descriptor: Joint angles (normalized) or end-effector position
    """

    def __init__(
        self,
        n_dof: int = 20,
        target_pos: Tuple[float, float] = (0.8, 0.0),
        use_highdim_descriptor: bool = True,
        obstacle: Optional[Tuple[float, float, float, float]] = (
            0.5,
            0.55,
            -0.25,
            0.25,
        ),
    ):
        """
        Initialize arm task.

        Args:
            n_dof: Number of joints (degrees of freedom)
            target_pos: Target (x, y) position
            use_highdim_descriptor: If True, descriptor = joint angles (n_dof dim)
                                   If False, descriptor = end-effector (2D)
            obstacle: (x_min, x_max, y_min, y_max) or None
        """
        self.n_dof = n_dof
        self.link_length = 1.0 / n_dof
        self.target_pos = np.array(target_pos)
        self.use_highdim_descriptor = use_highdim_descriptor
        self.obstacle = obstacle

        self._genome_dim = n_dof
        self._desc_dim = n_dof if use_highdim_descriptor else 2

    @property
    def genome_dim(self) -> int:
        return self._genome_dim

    @property
    def desc_dim(self) -> int:
        return self._desc_dim

    def forward_kinematics_batch(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute positions of all joints.

        Args:
            joints: Joint angles (batch_size, n_dof)

        Returns:
            Joint positions (batch_size, n_dof, 2)
        """
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)

    def check_collisions_batch(self, joint_coords: np.ndarray) -> np.ndarray:
        """
        Check collisions with obstacle.

        Args:
            joint_coords: Joint positions (batch_size, n_dof, 2)

        Returns:
            Boolean array (batch_size,) - True if collision
        """
        if self.obstacle is None:
            return np.zeros(len(joint_coords), dtype=bool)

        x_min, x_max, y_min, y_max = self.obstacle
        batch_size = joint_coords.shape[0]

        # Add origin
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)

        # Check point inside box
        p_in_x = (points[:, :, 0] > x_min) & (points[:, :, 0] < x_max)
        p_in_y = (points[:, :, 1] > y_min) & (points[:, :, 1] < y_max)
        any_inside = np.any(p_in_x & p_in_y, axis=1)

        # Check line segment intersections
        A, B = points[:, :-1, :], points[:, 1:, :]
        Ax, Ay, Bx, By = A[:, :, 0], A[:, :, 1], B[:, :, 0], B[:, :, 1]
        dx, dy = Bx - Ax, By - Ay

        # Avoid division by zero
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)

        def check_vertical(wall_x):
            t = (wall_x - Ax) / dx
            y_at_t = Ay + t * dy
            return (t >= 0) & (t <= 1) & (y_at_t >= y_min) & (y_at_t <= y_max)

        def check_horizontal(wall_y):
            t = (wall_y - Ay) / dy
            x_at_t = Ax + t * dx
            return (t >= 0) & (t <= 1) & (x_at_t >= x_min) & (x_at_t <= x_max)

        hit = (
            check_vertical(x_min)
            | check_vertical(x_max)
            | check_horizontal(y_min)
            | check_horizontal(y_max)
        )

        return any_inside | np.any(hit, axis=1)

    def evaluate(self, genomes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate batch of genomes.

        Args:
            genomes: Joint angles (batch_size, n_dof)

        Returns:
            fitness: Fitness values (batch_size,)
            descriptors: Behavior descriptors (batch_size, desc_dim)
        """
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]

        # Fitness: 1 - distance to target
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = np.maximum(1.0 - dists, 0.0)

        # Collision penalty
        collisions = self.check_collisions_batch(joint_coords)
        fitness[collisions] = 0.0

        # Descriptor
        if self.use_highdim_descriptor:
            # Normalize to [0, 1]
            descriptors = (genomes + np.pi) / (2 * np.pi)
        else:
            # End-effector descriptor (normalized to approx range)
            # Arm length is 1.0, so tip is in circle of radius 1.0
            # Normalize from [-1, 1] to [0, 1]
            descriptors = (tips + 1.0) / 2.0
            descriptors = np.clip(descriptors, 0, 1)

        return fitness, descriptors
