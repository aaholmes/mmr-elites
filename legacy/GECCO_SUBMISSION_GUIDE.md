# MUSE-QD: Path to GECCO Submission

## Current Progress Assessment

### ✅ What You've Implemented (Great Progress!)

| Component | Status | Quality |
|-----------|--------|---------|
| Rust MMR Selector | ✅ Complete | Excellent - lazy greedy with staleness tracking |
| PyO3 Bindings | ✅ Complete | Zero-copy working |
| Arm20 Task | ✅ Complete | Good obstacle avoidance |
| Ant-v4 Task | ✅ Complete | MuJoCo integration working |
| MAP-Elites Baseline | ✅ Complete | Sparse grid implementation |
| Metrics Module | ✅ Complete | QD-Score, coverage, uniformity |
| Visualization | ✅ Complete | Publication-quality plots |
| Experiment Runner | ✅ Complete | Multi-seed support |
| Unit Tests | ✅ Complete | Good coverage of metrics |
| Benchmark Script | ✅ Complete | Ready to run |

### 📊 Estimated Completion: ~65-70%

You've built all the **infrastructure**. What's missing is the **science**:
- Actual experimental results
- Statistical analysis
- Additional baselines (CVT-MAP-Elites)
- Ablation studies
- The paper itself

---

## Does MUSE-QD Stand a Chance Against MAP-Elites?

### Short Answer: **Yes, but with caveats.**

### The Nuanced Truth:

**Where MUSE-QD Should Win:**
1. **High-dimensional behavior spaces (D > 10)**: This is your core claim. MAP-Elites with 5 bins/dim on 20D = 5²⁰ ≈ 10¹⁴ cells. It degrades to random search.

2. **Fixed memory guarantee**: You always have exactly K elites. MAP-Elites archive size is unpredictable.

3. **Uniform coverage**: MMR explicitly optimizes for spread. MAP-Elites has grid artifacts.

**Where MAP-Elites Might Win:**
1. **Low-dimensional spaces (D ≤ 6)**: Grids work fine here. MUSE-QD's O(K log K) selection is overhead.

2. **Interpretability**: Grid cells have semantic meaning (e.g., "fast + left" vs "slow + right").

3. **Established track record**: Reviewers know MAP-Elites. Novel methods get scrutinized.

### The Key Experiment You MUST Run:

```
Task: 20-DOF Arm (using JOINT ANGLES as behavior descriptor, not end-effector!)
```

This is **critical**. Your current `benchmark_arm20.py` uses 2D end-effector position as descriptor. That's **too easy** for MAP-Elites (only 2D grid needed).

Change the behavior descriptor to the **20 joint angles themselves** to demonstrate the curse of dimensionality.

---

## What's Missing for GECCO

### Critical Missing Pieces

| Item | Importance | Effort | Status |
|------|------------|--------|--------|
| Run experiments with 20D descriptors | **CRITICAL** | 2 hrs | ❌ |
| CVT-MAP-Elites baseline | **HIGH** | 4-6 hrs | ❌ |
| λ ablation study | **HIGH** | 2-3 hrs | ❌ |
| Statistical significance tests | **HIGH** | 2 hrs | ❌ |
| Archive size K ablation | Medium | 2 hrs | ❌ |
| Runtime comparison plot | Medium | 1 hr | ❌ |
| Write the paper | **CRITICAL** | 20-30 hrs | ❌ |

---

## Full Implementation Guide: Critical Next Steps

### Step 1: Fix the Behavior Descriptor (CRITICAL)

Your current code uses 2D end-effector position. For the paper's main claim to work, you need 20D joint angles.

**Create this new task file:**

```python
# tasks/arm_20_highdim.py
"""
20-DOF Arm with HIGH-DIMENSIONAL behavior descriptor.
This is the CRITICAL task for demonstrating MUSE-QD's advantage.
"""

import numpy as np

class Arm20HighDimTask:
    """
    20-DOF Arm where behavior descriptor = joint angles (20D).
    
    This is where MAP-Elites FAILS and MUSE-QD WINS.
    """
    
    def __init__(self, target_pos=(0.8, 0.0)):
        self.dof = 20
        self.link_length = 1.0 / self.dof
        self.target_pos = np.array(target_pos)
        
        # Obstacle
        self.box_x = [0.5, 0.55]
        self.box_y = [-0.25, 0.25]
    
    def forward_kinematics_batch(self, joints):
        angles = np.cumsum(joints, axis=1)
        dx = self.link_length * np.cos(angles)
        dy = self.link_length * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)
    
    def check_collisions_batch(self, joint_coords):
        batch_size = joint_coords.shape[0]
        origin = np.zeros((batch_size, 1, 2))
        points = np.concatenate([origin, joint_coords], axis=1)
        
        p_in_x = (points[:, :, 0] > self.box_x[0]) & (points[:, :, 0] < self.box_x[1])
        p_in_y = (points[:, :, 1] > self.box_y[0]) & (points[:, :, 1] < self.box_y[1])
        any_inside = np.any(p_in_x & p_in_y, axis=1)
        
        A = points[:, :-1, :]
        B = points[:, 1:, :]
        Ax, Ay = A[:, :, 0], A[:, :, 1]
        Bx, By = B[:, :, 0], B[:, :, 1]
        dx, dy = Bx - Ax, By - Ay
        dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
        dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
        
        def check_vertical(wall_x, y_min, y_max):
            t = (wall_x - Ax) / dx
            y_at_t = Ay + t * dy
            return (t >= 0) & (t <= 1) & (y_at_t >= y_min) & (y_at_t <= y_max)
        
        def check_horizontal(wall_y, x_min, x_max):
            t = (wall_y - Ay) / dy
            x_at_t = Ax + t * dx
            return (t >= 0) & (t <= 1) & (x_at_t >= x_min) & (x_at_t <= x_max)
        
        hit = (
            check_vertical(self.box_x[0], self.box_y[0], self.box_y[1]) |
            check_vertical(self.box_x[1], self.box_y[0], self.box_y[1]) |
            check_horizontal(self.box_y[0], self.box_x[0], self.box_x[1]) |
            check_horizontal(self.box_y[1], self.box_x[0], self.box_x[1])
        )
        return any_inside | np.any(hit, axis=1)
    
    def evaluate(self, genomes):
        """
        Returns:
            fitness: Distance-based fitness (N,)
            descriptors: JOINT ANGLES as behavior descriptor (N, 20)
        """
        joint_coords = self.forward_kinematics_batch(genomes)
        tips = joint_coords[:, -1, :]
        
        dists = np.linalg.norm(tips - self.target_pos, axis=1)
        fitness = 1.0 - dists
        fitness = np.maximum(fitness, 0.0)
        
        collides = self.check_collisions_batch(joint_coords)
        fitness[collides] = 0.0
        
        # KEY DIFFERENCE: Return joint angles as descriptor (20D)
        # Normalize to [0, 1] for better distance computation
        descriptors = (genomes + np.pi) / (2 * np.pi)
        
        return fitness, descriptors
```

### Step 2: Implement CVT-MAP-Elites Baseline

CVT-MAP-Elites is the **main competitor** for unstructured archives. You MUST compare against it.

```python
# baselines/cvt_map_elites.py
"""
CVT-MAP-Elites: Centroidal Voronoi Tessellation MAP-Elites

Reference: Vassiliades et al. (2016) "Using Centroidal Voronoi Tessellations 
to Scale Up the Multi-dimensional Archive of Phenotypic Elites Algorithm"
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

class CVTArchive:
    """
    Archive based on Centroidal Voronoi Tessellation.
    
    Instead of a grid, uses K pre-computed centroids.
    Each individual is assigned to nearest centroid.
    """
    
    def __init__(
        self, 
        n_niches: int,
        descriptor_dim: int,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        seed: int = 42
    ):
        self.n_niches = n_niches
        self.descriptor_dim = descriptor_dim
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        
        # Generate CVT centroids using k-means on uniform samples
        self.centroids = self._compute_cvt_centroids(seed)
        self.tree = cKDTree(self.centroids)
        
        # Archive storage: index -> (genome, fitness, descriptor)
        self.archive = {}
    
    def _compute_cvt_centroids(self, seed: int, n_samples: int = 100000) -> np.ndarray:
        """Compute CVT centroids using Lloyd's algorithm."""
        rng = np.random.default_rng(seed)
        
        # Initialize with random samples
        samples = rng.uniform(
            self.bounds_min, 
            self.bounds_max, 
            size=(n_samples, self.descriptor_dim)
        )
        
        # K-means to find centroids
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_niches, random_state=seed, n_init=1)
        kmeans.fit(samples)
        
        return kmeans.cluster_centers_
    
    def get_niche(self, descriptor: np.ndarray) -> int:
        """Find the nearest centroid for a descriptor."""
        _, idx = self.tree.query(descriptor)
        return idx
    
    def add(self, genome: np.ndarray, fitness: float, descriptor: np.ndarray) -> bool:
        """
        Try to add individual to archive.
        Returns True if added/replaced, False otherwise.
        """
        niche = self.get_niche(descriptor)
        
        if niche not in self.archive or fitness > self.archive[niche][1]:
            self.archive[niche] = (genome.copy(), fitness, descriptor.copy())
            return True
        return False
    
    def sample_parents(self, n: int) -> np.ndarray:
        """Sample n parents uniformly from archive."""
        if not self.archive:
            return None
        
        keys = list(self.archive.keys())
        indices = np.random.randint(0, len(keys), n)
        return np.array([self.archive[keys[i]][0] for i in indices])
    
    def get_all_fitness(self) -> np.ndarray:
        return np.array([v[1] for v in self.archive.values()])
    
    def get_all_descriptors(self) -> np.ndarray:
        return np.array([v[2] for v in self.archive.values()])
    
    def __len__(self):
        return len(self.archive)


def run_cvt_map_elites(
    task,
    n_niches: int,
    generations: int,
    batch_size: int,
    mutation_sigma: float,
    seed: int,
    log_interval: int = 50,
) -> dict:
    """Run CVT-MAP-Elites algorithm."""
    from metrics import compute_all_metrics
    
    np.random.seed(seed)
    
    # Infer descriptor bounds from task (you may need to adjust)
    # For normalized joint angles: [0, 1]^20
    descriptor_dim = 20
    bounds_min = np.zeros(descriptor_dim)
    bounds_max = np.ones(descriptor_dim)
    
    archive = CVTArchive(
        n_niches=n_niches,
        descriptor_dim=descriptor_dim,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        seed=seed
    )
    
    # Initialize
    init_pop = np.random.uniform(-np.pi, np.pi, (batch_size * 5, 20))
    fit, desc = task.evaluate(init_pop)
    
    for i in range(len(init_pop)):
        archive.add(init_pop[i], fit[i], desc[i])
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "archive_size": []}
    
    import time
    start_time = time.time()
    
    for gen in range(1, generations + 1):
        # Sample parents
        parents = archive.sample_parents(batch_size)
        if parents is None:
            continue
        
        # Mutate
        offspring = parents + np.random.normal(0, mutation_sigma, parents.shape)
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        # Evaluate and add
        off_fit, off_desc = task.evaluate(offspring)
        for i in range(len(offspring)):
            archive.add(offspring[i], off_fit[i], off_desc[i])
        
        # Log
        if gen % log_interval == 0:
            all_fit = archive.get_all_fitness()
            all_desc = archive.get_all_descriptors()
            metrics = compute_all_metrics(all_fit, all_desc)
            
            history["generation"].append(gen)
            history["qd_score"].append(metrics["qd_score"])
            history["max_fitness"].append(metrics["max_fitness"])
            history["mean_pairwise_distance"].append(metrics["mean_pairwise_distance"])
            history["archive_size"].append(len(archive))
    
    runtime = time.time() - start_time
    
    all_fit = archive.get_all_fitness()
    all_desc = archive.get_all_descriptors()
    final_metrics = compute_all_metrics(all_fit, all_desc)
    final_metrics["archive_size"] = len(archive)
    
    return {
        "algorithm": "CVT-MAP-Elites",
        "seed": seed,
        "runtime": runtime,
        "final_metrics": final_metrics,
        "history": history,
    }
```

### Step 3: Lambda Ablation Study Script

```python
# experiments/lambda_ablation.py
"""
Lambda Ablation Study for MUSE-QD

Tests λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
"""

import numpy as np
import json
from pathlib import Path
import mmr_elites_rs
from tasks.arm_20_highdim import Arm20HighDimTask
from metrics import compute_all_metrics

LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
N_SEEDS = 5
GENERATIONS = 1000
ARCHIVE_SIZE = 1000
BATCH_SIZE = 200
MUTATION_SIGMA = 0.1

def run_ablation():
    output_dir = Path("results/lambda_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = Arm20HighDimTask()
    results = {}
    
    for lambda_val in LAMBDA_VALUES:
        print(f"\n{'='*50}")
        print(f"Testing λ = {lambda_val}")
        print('='*50)
        
        results[lambda_val] = []
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=" ", flush=True)
            
            np.random.seed(seed)
            selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, lambda_val)
            
            archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
            fit, desc = task.evaluate(archive)
            
            indices = selector.select(fit, desc)
            archive = archive[indices]
            fit = fit[indices]
            desc = desc[indices]
            
            for gen in range(1, GENERATIONS + 1):
                parents = archive[np.random.randint(0, len(archive), BATCH_SIZE)]
                offspring = parents + np.random.normal(0, MUTATION_SIGMA, (BATCH_SIZE, 20))
                offspring = np.clip(offspring, -np.pi, np.pi)
                
                off_fit, off_desc = task.evaluate(offspring)
                
                pool_genes = np.vstack([archive, offspring])
                pool_fit = np.concatenate([fit, off_fit])
                pool_desc = np.vstack([desc, off_desc])
                
                survivor_idx = selector.select(pool_fit, pool_desc)
                
                archive = pool_genes[survivor_idx]
                fit = pool_fit[survivor_idx]
                desc = pool_desc[survivor_idx]
            
            metrics = compute_all_metrics(fit, desc)
            results[lambda_val].append(metrics)
            print(f"QD: {metrics['qd_score']:.2f}, MaxFit: {metrics['max_fitness']:.4f}")
    
    # Aggregate results
    summary = {}
    for lam in LAMBDA_VALUES:
        qd_scores = [r["qd_score"] for r in results[lam]]
        max_fits = [r["max_fitness"] for r in results[lam]]
        diversities = [r["mean_pairwise_distance"] for r in results[lam]]
        
        summary[str(lam)] = {
            "qd_score": {"mean": np.mean(qd_scores), "std": np.std(qd_scores)},
            "max_fitness": {"mean": np.mean(max_fits), "std": np.std(max_fits)},
            "diversity": {"mean": np.mean(diversities), "std": np.std(diversities)},
        }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    
    # Print table
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'λ':<8} | {'QD-Score':<20} | {'Max Fitness':<20} | {'Diversity':<20}")
    print("-"*70)
    for lam in LAMBDA_VALUES:
        s = summary[str(lam)]
        print(f"{lam:<8.1f} | {s['qd_score']['mean']:>8.2f}±{s['qd_score']['std']:<8.2f} | "
              f"{s['max_fitness']['mean']:>8.4f}±{s['max_fitness']['std']:<8.4f} | "
              f"{s['diversity']['mean']:>8.4f}±{s['diversity']['std']:<8.4f}")

if __name__ == "__main__":
    run_ablation()
```

### Step 4: Main Comparison Experiment (20D Descriptors)

```python
# experiments/main_comparison.py
"""
Main Experiment: MUSE-QD vs MAP-Elites vs CVT-MAP-Elites
on 20-DOF Arm with 20D behavior descriptors.

This is THE experiment for the paper.
"""

import numpy as np
import json
import pickle
from pathlib import Path
import time

import mmr_elites_rs
from tasks.arm_20_highdim import Arm20HighDimTask
from baselines.cvt_map_elites import run_cvt_map_elites
from metrics import compute_all_metrics

# Configuration
N_SEEDS = 10
GENERATIONS = 2000
ARCHIVE_SIZE = 1000  # For MUSE-QD and CVT
BATCH_SIZE = 200
MUTATION_SIGMA = 0.1
LAMBDA = 0.5

def run_muse_qd_20d(task, seed):
    np.random.seed(seed)
    selector = mmr_elites_rs.MMRSelector(ARCHIVE_SIZE, LAMBDA)
    
    archive = np.random.uniform(-np.pi, np.pi, (ARCHIVE_SIZE, 20))
    fit, desc = task.evaluate(archive)
    
    indices = selector.select(fit, desc)
    archive, fit, desc = archive[indices], fit[indices], desc[indices]
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": []}
    
    start = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        parents = archive[np.random.randint(0, len(archive), BATCH_SIZE)]
        offspring = parents + np.random.normal(0, MUTATION_SIGMA, (BATCH_SIZE, 20))
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        off_fit, off_desc = task.evaluate(offspring)
        
        pool = np.vstack([archive, offspring])
        pool_fit = np.concatenate([fit, off_fit])
        pool_desc = np.vstack([desc, off_desc])
        
        idx = selector.select(pool_fit, pool_desc)
        archive, fit, desc = pool[idx], pool_fit[idx], pool_desc[idx]
        
        if gen % 100 == 0:
            m = compute_all_metrics(fit, desc)
            history["generation"].append(gen)
            history["qd_score"].append(m["qd_score"])
            history["max_fitness"].append(m["max_fitness"])
            history["mean_pairwise_distance"].append(m["mean_pairwise_distance"])
    
    final = compute_all_metrics(fit, desc)
    return {"algorithm": "MUSE-QD", "seed": seed, "runtime": time.time()-start,
            "final_metrics": final, "history": history}


def run_sparse_map_elites_20d(task, seed, bins_per_dim=3):
    """
    Sparse MAP-Elites on 20D descriptor space.
    With bins_per_dim=3, there are 3^20 ≈ 3.5 billion cells.
    Most cells will be empty → degrades to random search.
    """
    np.random.seed(seed)
    
    def get_cell(desc):
        # desc is in [0, 1]^20
        indices = (desc * bins_per_dim).astype(int)
        indices = np.clip(indices, 0, bins_per_dim - 1)
        return tuple(indices)
    
    archive = {}
    
    init_pop = np.random.uniform(-np.pi, np.pi, (BATCH_SIZE * 5, 20))
    fit, desc = task.evaluate(init_pop)
    
    for i in range(len(init_pop)):
        cell = get_cell(desc[i])
        if cell not in archive or fit[i] > archive[cell][1]:
            archive[cell] = (init_pop[i].copy(), fit[i], desc[i].copy())
    
    history = {"generation": [], "qd_score": [], "max_fitness": [], 
               "mean_pairwise_distance": [], "archive_size": []}
    
    start = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        keys = list(archive.keys())
        parent_keys = [keys[np.random.randint(len(keys))] for _ in range(BATCH_SIZE)]
        parents = np.array([archive[k][0] for k in parent_keys])
        
        offspring = parents + np.random.normal(0, MUTATION_SIGMA, parents.shape)
        offspring = np.clip(offspring, -np.pi, np.pi)
        
        off_fit, off_desc = task.evaluate(offspring)
        
        for i in range(len(offspring)):
            cell = get_cell(off_desc[i])
            if cell not in archive or off_fit[i] > archive[cell][1]:
                archive[cell] = (offspring[i].copy(), off_fit[i], off_desc[i].copy())
        
        if gen % 100 == 0:
            all_fit = np.array([v[1] for v in archive.values()])
            all_desc = np.array([v[2] for v in archive.values()])
            m = compute_all_metrics(all_fit, all_desc)
            
            history["generation"].append(gen)
            history["qd_score"].append(m["qd_score"])
            history["max_fitness"].append(m["max_fitness"])
            history["mean_pairwise_distance"].append(m["mean_pairwise_distance"])
            history["archive_size"].append(len(archive))
    
    all_fit = np.array([v[1] for v in archive.values()])
    all_desc = np.array([v[2] for v in archive.values()])
    final = compute_all_metrics(all_fit, all_desc)
    final["archive_size"] = len(archive)
    
    return {"algorithm": "MAP-Elites", "seed": seed, "runtime": time.time()-start,
            "final_metrics": final, "history": history}


def main():
    output_dir = Path("results/main_comparison_20d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task = Arm20HighDimTask()
    
    all_results = {
        "MUSE-QD": [],
        "MAP-Elites": [],
        "CVT-MAP-Elites": [],
    }
    
    print("="*60)
    print("MAIN EXPERIMENT: 20D Behavior Descriptor")
    print("="*60)
    
    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")
        
        print("  MUSE-QD...", end=" ", flush=True)
        r = run_muse_qd_20d(task, seed)
        all_results["MUSE-QD"].append(r)
        print(f"QD={r['final_metrics']['qd_score']:.2f}")
        
        print("  MAP-Elites...", end=" ", flush=True)
        r = run_sparse_map_elites_20d(task, seed)
        all_results["MAP-Elites"].append(r)
        print(f"QD={r['final_metrics']['qd_score']:.2f}, Archive={r['final_metrics']['archive_size']}")
        
        print("  CVT-MAP-Elites...", end=" ", flush=True)
        r = run_cvt_map_elites(task, ARCHIVE_SIZE, GENERATIONS, BATCH_SIZE, 
                               MUTATION_SIGMA, seed)
        all_results["CVT-MAP-Elites"].append(r)
        print(f"QD={r['final_metrics']['qd_score']:.2f}")
    
    # Save
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    metrics = ["qd_score", "max_fitness", "mean_pairwise_distance", "archive_size"]
    
    print(f"\n{'Metric':<25} | {'MUSE-QD':<18} | {'MAP-Elites':<18} | {'CVT-MAP-Elites':<18}")
    print("-"*85)
    
    for m in metrics:
        row = f"{m:<25} |"
        for alg in ["MUSE-QD", "MAP-Elites", "CVT-MAP-Elites"]:
            vals = [r["final_metrics"].get(m, 0) for r in all_results[alg]]
            row += f" {np.mean(vals):>7.2f}±{np.std(vals):<6.2f} |"
        print(row)
    
    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
```

### Step 5: Statistical Significance Tests

```python
# analysis/statistical_tests.py
"""
Statistical significance testing for QD experiments.
Uses Mann-Whitney U test (non-parametric, recommended for small samples).
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path
import pickle

def mann_whitney_test(group1, group2):
    """
    Mann-Whitney U test for comparing two groups.
    Returns U statistic and p-value.
    """
    statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return statistic, pvalue

def cohens_d(group1, group2):
    """Effect size using Cohen's d."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_results(results_file: str):
    """Run statistical tests on experiment results."""
    
    with open(results_file, "rb") as f:
        all_results = pickle.load(f)
    
    algorithms = list(all_results.keys())
    metrics = ["qd_score", "max_fitness", "mean_pairwise_distance"]
    
    print("="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    for metric in metrics:
        print(f"\n--- {metric.upper()} ---")
        
        # Extract values for each algorithm
        values = {}
        for alg in algorithms:
            values[alg] = [r["final_metrics"][metric] for r in all_results[alg]]
        
        # Pairwise comparisons
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                u_stat, p_val = mann_whitney_test(values[alg1], values[alg2])
                d = cohens_d(values[alg1], values[alg2])
                
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                print(f"  {alg1} vs {alg2}:")
                print(f"    Mean: {np.mean(values[alg1]):.4f} vs {np.mean(values[alg2]):.4f}")
                print(f"    Mann-Whitney U: {u_stat:.1f}, p={p_val:.4f} {sig}")
                print(f"    Cohen's d: {d:.3f}")

if __name__ == "__main__":
    analyze_results("results/main_comparison_20d/all_results.pkl")
```

---

## Timeline to GECCO Submission

| Week | Tasks | Hours |
|------|-------|-------|
| **Week 1** | Fix 20D descriptor, run main experiment | 8 |
| **Week 1** | Implement CVT-MAP-Elites | 6 |
| **Week 1** | Lambda ablation | 3 |
| **Week 2** | Statistical analysis | 2 |
| **Week 2** | Generate all figures | 4 |
| **Week 2** | Write paper draft | 20 |
| **Week 3** | Revisions, polish | 10 |
| **Total** | | ~53 hours |

---

## Paper Structure Outline

1. **Abstract** (150 words)
2. **Introduction** (1 page)
   - QD is important
   - MAP-Elites fails in high-D
   - We propose MUSE-QD
3. **Background** (0.5 page)
   - MAP-Elites
   - MMR from IR
4. **Method** (1.5 pages)
   - MMR objective
   - Lazy greedy algorithm
   - Complexity analysis
5. **Experiments** (2 pages)
   - 20-DOF Arm (main result)
   - Lambda ablation
   - Runtime comparison
6. **Results** (1 page)
   - Tables with stats
   - Learning curves
7. **Discussion & Conclusion** (0.5 page)

---

## Key Takeaways

1. **You're ~65% done** - infrastructure is solid
2. **The critical gap**: Run experiments with 20D descriptors (not 2D)
3. **Must-have baseline**: CVT-MAP-Elites
4. **Your algorithm WILL win** on high-D if you set up the experiment correctly
5. **Start writing NOW** - don't wait until experiments are done

Good luck! 🚀
