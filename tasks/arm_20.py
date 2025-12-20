import numpy as np

class Arm20Task:
    def __init__(self, target_pos=(0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof
        self.target_pos = np.array(target_pos)
        
        # Wall AABB (Axis-Aligned Bounding Box)
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]

    def forward_kinematics_batch(self, joints):
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)

    def evaluate(self, genomes):
        batch_size = genomes.shape[0]
        
        # 1. Kinematics
        # joint_coords: (Batch, 20, 2)
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]
        
        # 2. Fitness
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = 1.0 - dists
        fitness = np.maximum(fitness, 0.0)
        
        # 3. EXACT COLLISION DETECTION
        # Define segments: A -> B
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)
        A = points[:, :-1, :] # Start points (Batch, 20, 2)
        B = points[:, 1:, :]  # End points   (Batch, 20, 2)
        
        # --- Test 1: Points Inside (Catch fully contained links) ---
        # Only need to check A (B is A of next link) + last B
        # But for vectorized speed, just check all points.
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_point_inside = np.any(p_in_x & p_in_y, axis=1)

        # --- Test 2: Line Segment Intersections ---
        # We check intersection with the 4 infinite lines defining the box.
        # If intersect occurs at 0<=t<=1 AND within the perpendicular bounds, it's a hit.
        
        # Vector helpers
        Ax, Ay = A[:,:,0], A[:,:,1]
        Bx, By = B[:,:,0], B[:,:,1]
        dx, dy = Bx - Ax, By - Ay
        
        # Avoid division by zero
        dx[np.abs(dx) < 1e-9] = 1e-9
        dy[np.abs(dy) < 1e-9] = 1e-9

        # Function to check vertical wall (x = constant)
        # Returns Boolean matrix (Batch, 20)
        def check_vertical(wall_x, y_min, y_max):
            t = (wall_x - Ax) / dx
            y_at_t = Ay + t * dy
            return (t >= 0) & (t <= 1) & (y_at_t >= y_min) & (y_at_t <= y_max)

        # Function to check horizontal wall (y = constant)
        def check_horizontal(wall_y, x_min, x_max):
            t = (wall_y - Ay) / dy
            x_at_t = Ax + t * dx
            return (t >= 0) & (t <= 1) & (x_at_t >= x_min) & (x_at_t <= x_max)

        # Check all 4 boundaries
        hit_left   = check_vertical(self.box_x[0], self.box_y[0], self.box_y[1])
        hit_right  = check_vertical(self.box_x[1], self.box_y[0], self.box_y[1])
        hit_bottom = check_horizontal(self.box_y[0], self.box_x[0], self.box_x[1])
        hit_top    = check_horizontal(self.box_y[1], self.box_x[0], self.box_x[1])
        
        any_intersection = np.any(hit_left | hit_right | hit_bottom | hit_top, axis=1)
        
        # Final Collision = Point Inside OR Line Intersect
        collides = any_point_inside | any_intersection
        
        fitness[collides] = 0.0
        
        return fitness, tips

    def get_descriptor(self, genomes):
        _, tips = self.evaluate(genomes)
        return tips 
