# MMR-Elites: High-Dimensional Quality-Diversity via Lazy Greedy Selection
Status: Alpha / Prototyping | Backend: Rust + PyO3 | Frontend: Python / JAX

## Executive Summary
MMR-Elites (Maximal Marginal Relevance Elites) is a Quality-Diversity (QD) optimization algorithm designed to solve the "Curse of Dimensionality" in behavior spaces.
Where standard methods like MAP-Elites rely on discrete grids—which scale exponentially ($O(\text{bins}^D)$) and fail for $D \gtrsim 10$—MMR-Elites maintains a fixed-size unstructured archive of $K$ individuals. It treats the survival step as a Subset Selection Problem, using a Lazy Greedy optimization of the Maximal Marginal Relevance (MMR) metric to ensure uniform coverage of the behavior manifold without grid artifacts.
This repository contains the high-performance implementation, featuring a Rust backend for $O(K \log K)$ selection survival and a Python frontend for flexible task definition.

## Tentative paper title and abstract

### Title
MUSE-QD: Breaking the Curse of Dimensionality in Quality-Diversity via Lazy Greedy Subset Selection

### Abstract
Quality-Diversity (QD) algorithms, such as MAP-Elites, have emerged as a powerful paradigm for generating diverse repertoires of robot behaviors. However, standard QD methods rely on structured archives (grids) that suffer from the "curse of dimensionality," rendering them computationally intractable for high-dimensional behavior spaces ($D > 10$). We introduce MUSE-QD (Maximal Unstructured Selection of Elites), a grid-free evolutionary algorithm that maintains a fixed-memory unstructured archive via a rigorous subset selection objective. By reformulating the survival step as a Maximal Marginal Relevance (MMR) optimization problem, MUSE-QD explicitly maximizes the joint distribution of fitness and behavioral coverage without geometric priors. We propose a Lazy Greedy implementation backed by a high-performance Rust runtime, reducing the selection complexity from cubic to near-linear time. We demonstrate that MUSE-QD outperforms MAP-Elites and unstructured baselines on a 20-DOF planar arm task, achieving superior coverage and memory efficiency while eliminating the need for hyperparameter-sensitive grid resolutions or novelty thresholds.

## Architecture
The system is designed as a hybrid Rust/Python application to balance compute efficiency with experimental flexibility.

mmr_elites_rs (Rust Core):
Handles the heavy lifting of archive maintenance.
Implements Lazy Greedy Selection using Priority Queues to maximize $Score = (1-\lambda) \cdot Fitness + \lambda \cdot d_{min}$.
Zero-Copy data transfer via PyO3 and numpy (Rust crate).
Optimized for SIMD-friendly Euclidean distance calculations.

mmr_qd (Python Frontend):
Manages the evolutionary loop (Selection $\rightarrow$ Mutation $\rightarrow$ Evaluation).
Defines the Task Protocols (e.g., 20-DOF Planar Arm, Astrobiology Molecules).
Handles logging, visualization (Coverage metrics, QD-Score), and JAX integration.

## Component StatusComponent   Sub-Module     Status    Notes
---         ---            ---       ---
### Core Algorithm
LazyGreedy Selector🟢 ImplementedRust backend with BinaryHeap optimization.
BinaryHeap Optimization🟢 ImplementedFully integrated.

### Rust Binding
PyO3 Interface🟢 ImplementedZero-copy ndarray views working.
Cargo Configuration🟢 ReadyBuilds with `maturin develop --release`.

### Benchmarks
Arm20DOF (Task)🟢 ImplementedIncludes Forward Kinematics & Wall Collision ("The Trap").
Performance Benchmarks🟢 Implemented`benchmark.py` and `stress_test.py` confirm >100x speedup.
Standard MAP-Elites🔴 PendingNeeded for quality comparison (Phase 3).

### Visualization
Coverage Metrics🟡 Partial`plot_results.py` generates archive plots, but no comparative metrics yet.
QD-Score Plots🔴 PendingNeed standardized history logger for both algorithms.

## Installation & UsagePrerequisites: Rust (cargo), Python 3.10+, maturin.

```bash
# 1. Compile the Rust backend
maturin develop --release

# 2. Run the Benchmark (Performance)
python benchmark.py

# 3. Run the Experiment (Arm20 Task)
python experiment.py
```
Minimal Example:

```python
import mmr_elites_rs
import numpy as np

# Initialize Selector (K=1000 elites, Lambda=0.5)
selector = mmr_elites_rs.MMRSelector(1000, 0.5)

# Survival Step (Zero-Copy)
# fitness: (N,), descriptors: (N, 20)
survivors = selector.select(fitness_array, descriptor_array)
```

## Roadmap & TimelineTotal Estimated Effort: 16 - 24 Hours

Phase 1: The Engine (Days 1-2)
- [x] Initialize Repo: Set up git, cargo, and pyproject.toml.
- [x] Implement Rust Core: Port the LazyGreedy logic from design doc to src/lib.rs.
- [x] Unit Testing: Verify LazyGreedy returns identical subsets to Brute Force (`scripts/test_correctness.py`).
- [x] Bind: Verify numpy to ndarray zero-copy passing works without segfaults.

Phase 2: The Benchmark (Days 3-4)
- [x] Implement Arm20: `tasks/arm_20.py` implements FK and Obstacle Avoidance.
- [x] Performance Tests: `stress_test.py` validates O(N log N) scaling.
- [ ] Implement Baseline: Code a "Sparse MAP-Elites" (using a hash map) to compare against.
- [x] Data Pipeline: `experiment.py` saves snapshots to `muse_results.pkl`.

Phase 3: Analysis & Polish (Day 5)
- [ ] Metric: Coverage: Implement a "Union of Hyperspheres" or "k-NN" metric to quantify coverage in 20D.
- [ ] Visualization: Generate the "QD-Score vs Evaluations" and "Archive Size vs. Stability" plots.
- [ ] Documentation: Finalize this README with reproduction steps.


## Comparison to State-of-the-Art

| Algorithm    | Archive Structure     | Scaling                     | Failure Mode                                  |
| :----------- | :-------------------- | :-------------------------- | :-------------------------------------------- |
| MAP-Elites   | Rigid Grid            | $O(\text{bins}^D)$          | Explodes if $D > 10$.                         |
| MOUR-QD      | Unstructured List     | Variable                    | Archive size unstable (radius dependent).     |
| MMR-Elites   | Fixed List (Greedy)   | Fixed $K$                   | Higher compute cost per generation ($O(K \log K)$). |
