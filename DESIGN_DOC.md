# Design Document: MMR-Elites Implementation

## System Architecture
The system follows a hybrid architecture to balance performance with ease of experimentation.

Rust Backend (mmr_elites_rs): Handles the "Survival Step." It manages the unstructured archive, computes Euclidean distances, and executes the Greedy Subset Selection loop. This is critical because the selection logic is $O(K^2)$ or $O(K \log K)$ and requires heavy array operations.

### The Individual
A struct representing a single solution.

```rust
struct Individual {
    id: u64,
    fitness: f64,
    descriptor: Vec<f64>, // The behavior vector 'b'
    genes: Vec<f64>,      // The genotype (optional, can be kept in Python to save RAM)
}
```

Python Frontend (mmr_qd): Manages the evolutionary loop (generation of offspring), interacts with the simulation (fitness/descriptor evaluation), and handles data logging/visualization.

### The Unstructured Archive
Instead of a multi-dimensional array (Grid), this is a flat vector of fixed capacity $K$.

```rust
struct MMRArchive {
    capacity: usize, // K
    elites: Vec<Individual>,
    lambda: f64, // Mixing parameter for MMR (usually 0.5)
}
```

## Algorithm Logic: The Survival StepThe core innovation is replacing grid contention with a Greedy Subset Selection loop.

### The "Lazy Greedy" Optimization
A naive implementation of the selection loop scales poorly ($O(K \cdot M \cdot D)$). To make this tractable, we must implement the Lazy Greedy optimization using a Priority Queue, as $d_{min}$ (distance to the archive) is "submodular" (it can only decrease as items are added to the archive).

**Logic Flow:**
*   Input: A pool $\mathcal{P}$ (Current Archive + New Offspring)
*   Initialization: Create an empty next_archive.
*   Seed: Find $p^* \in \mathcal{P}$ with the highest fitness. Move $p^*$ to next_archive.
*   Priority Queue Construction:
    *   Calculate upper bounds for the scores of all remaining candidates.
    *   Push all candidates into a Max-Priority Queue.
*   Selection Loop (Iterate until size $K$):
    *   Pop the top candidate $c$ from the queue.
    *   Re-evaluate: Compute its actual marginal value (distance to the current next_archive).
    *   Check: If the re-evaluated score is still higher than the next item in the queue, select $c$ and add to next_archive.
    *   Else: Push $c$ back into the queue with the updated lower score (Lazy evaluation).

### Distance Metric
Strict adherence to the MaxMin metric is required. We maximize the minimum distance to any individual currently in the selected set. Avoid MaxSum: Do not maximize the sum of distances, as this pushes solutions to the boundaries (convex hull) and fails to cover the interior.## Python-Rust Interface (PyO3)
We will expose the Rust backend as a compiled Python module.

### API Definition:
```python
import mmr_elites_rs

class MMRSelector:
    def __init__(self, k: int, lambda_val: float = 0.5):
        """Initialize the selector with archive size K."""
        pass

    def select(self, candidates: list[dict]) -> list[int]:
        """
        Input: List of dicts {'id', 'fitness', 'descriptor'}
        Output: Indices of the survivors to keep.
        """
        pass
```
## Implementation Plan & Estimates
Total Estimated Time: 16 - 24 Hours

### Phase 1: Rust Core (4-6 hours)
*   Implement Individual and MMRArchive structs.
*   Implement brute-force MMR selection (for correctness testing).
*   Implement LazyGreedy optimization using std::collections::BinaryHeap.
*   Unit Test: Verify Lazy Greedy returns the exact same subset as Brute Force.

### Phase 2: Python Bindings (3-4 hours)
*   Set up maturin project structure.
*   Map Python lists/Numpy arrays to Rust Vec<f64>.
*   Handle memory safety (passing ownership vs. references).

### Phase 3: Python Evolutionary Loop (4-6 hours)
*   Implement the main loop: Selection -> Mutation -> Evaluation -> Union.
*   Implement a standard GaussianMutation operator.
*   Create abstract Task class.

### Phase 4: Benchmarks & Visualization (5-8 hours)
*   Implement the "20-DOF Planar Arm" task.
*   Implement a standard MAP-Elites grid (using a dictionary of tuples for sparse storage) for comparison.
*   Generate heatmaps (using PCA/t-SNE for high-dim visualization) and QD-Score plots.## Test Plan: MMR-Elites vs. MAP-Elites
To validate the claims in the document, you must prove that MMR-Elites works where MAP-Elites breaks.

### Test 1: The "Sanity Check" (Low Dimension)
*   **Domain:** 2D or 3D Point Navigation / Standard Arm Reaching.
*   **Configuration:** MAP-Elites (50x50 grid) vs. MMR-Elites ($K=2500$).
*   **Goal:** Confirm MMR-Elites achieves similar coverage and max fitness to standard methods in easy domains.

### Test 2: The "Killer App" (High Dimension)
*   **Domain:** 20-DOF Planar Arm Reaching.
*   **Behavior Descriptor:** The vector of all 20 joint angles ($\mathbb{R}^{20}$).
*   **Constraint:** This space is too large for a grid. Even 2 bins per dimension results in $10^6$ cells; 5 bins results in $9 \times 10^{13}$.
*   **Comparison:**
    *   MAP-Elites (Low Res): 2 bins per dimension. Expected result: Random search behavior due to aliasing (all points fall into the same few bins).
    *   MMR-Elites: Fixed $K=1000$ or $K=5000$.
*   **Metric:** "QD-Score" (Sum of fitness of all elites) and "Coverage" (Percentage of reachable workspace covered).
*   **Hypothesis:** MMR-Elites will form a diverse cloud of solutions; MAP-Elites will fail to expand.

### Test 3: Stability Comparison (vs MOUR-QD)
*   **Domain:** Unbounded continuous optimization (e.g., Rastrigin function).
*   **Configuration:** Run MOUR-QD with varying radii ($l$) and MMR-Elites with fixed $K$.
*   **Goal:** Demonstrate that MOUR-QD's archive size fluctuates unpredictably (explodes or collapses) depending on the radius, whereas MMR-Elites maintains exactly $K$ elites, satisfying the "Fixed Memory" constraint.
