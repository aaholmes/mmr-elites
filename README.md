# MMR-Elites: High-Dimensional Quality-Diversity via Lazy Greedy Selection
Status: Alpha / Prototyping | Backend: Rust + PyO3 | Frontend: Python / JAX

## Executive Summary
MMR-Elites (Maximal Marginal Relevance Elites) is a Quality-Diversity (QD) optimization algorithm designed to solve the "Curse of Dimensionality" in behavior spaces.
Where standard methods like MAP-Elites rely on discrete grids—which scale exponentially ($O(\text{bins}^D)$) and fail for $D \gtrsim 10$—MMR-Elites maintains a fixed-size unstructured archive of $K$ individuals. It treats the survival step as a Subset Selection Problem, using a Lazy Greedy optimization of the Maximal Marginal Relevance (MMR) metric to ensure uniform coverage of the behavior manifold without grid artifacts.
This repository contains the high-performance implementation, featuring a Rust backend for $O(K \log K)$ selection survival and a Python frontend for flexible task definition.

## Architecture
The system is designed as a hybrid Rust/Python application to balance compute efficiency with experimental flexibility.

mmr_rs (Rust Core):
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
LazyGreedy Selector🟡 DesignedLogic defined; needs unit tests for corner cases.
BinaryHeap Optimization🟡 DesignedImplemented in design doc; needs integration.

### Rust Binding
PyO3 Interface🟡 DesignedZero-copy ndarray views prototyped.
Cargo Configuration🟢 Readymaturin build specs defined.

### Benchmarks
Arm20DOF (Forward Kinematics)🟡 DesignedPython prototype ready; moving to JAX for speed?
Standard MAP-Elites🔴 PendingSparse dictionary implementation needed for baseline.

### Visualization
Coverage Metrics🔴 PendingNeed definition for "Continuous Coverage".
QD-Score Plots🔴 PendingStandard logging harness needed.

## Installation & UsagePrerequisites: Rust (cargo), Python 3.10+, maturin.

```bash
# 1. Compile the Rust backend
maturin develop --release

# 2. Run the Benchmark
python scripts/run_arm20_benchmark.py
```
Minimal Example:

```python
import mmr_rs
import numpy as np

# Initialize Selector (K=1000 elites, Lambda=0.5)
selector = mmr_rs.MuseSelector(1000, 0.5)

# Survival Step (Zero-Copy)
# fitness: (N,), descriptors: (N, 20)
survivors = selector.select(fitness_array, descriptor_array)
```

## Roadmap & TimelineTotal Estimated Effort: 16 - 24 Hours

Phase 1: The Engine (Days 1-2)
- [ ] Initialize Repo: Set up git, cargo, and pyproject.toml.
- [ ] Implement Rust Core: Port the LazyGreedy logic from design doc to src/lib.rs.
- [ ] Unit Testing: Verify LazyGreedy returns identical subsets to Brute Force for small $N$.
- [ ] Bind: Verify numpy to ndarray zero-copy passing works without segfaults.

Phase 2: The Benchmark (Days 3-4)
- [ ] Implement Arm20: Create the 20-DOF Arm task with redundant solutions.
- [ ] Implement Baseline: Code a "Sparse MAP-Elites" (using a hash map) to compare against.
- [ ] Data Pipeline: Create a standardized History logger for reproduction.

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
