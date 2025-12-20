# The Project Plan: "Operation MMR" (2-Week Sprint)

This is a "Stealth Mode" sprint. We strip away the fluff. We only build what is needed to generate The Plot.

## Phase 1: The "Transmission" (Rust Core) [Days 1-4]
**Goal**: A compiled Python module `mmr_elites_rs` that passes unit tests.

### Step 1.1: The Structs
*   Define `MMRSelector` struct in Rust.
*   It needs to hold the `archive_vectors` (2D Array) and `archive_fitness` (1D Array).

### Step 1.2: The "Lazy Greedy" Algorithm
**Crucial Detail**: You need a Min-Heap (Priority Queue) for the marginal gains.

**Algorithm**:
1.  Initialize Heap with upper-bound gains for all candidates.
2.  Pop best candidate.
3.  Re-evaluate its true gain (using the current selected set).
4.  If the new gain is still better than the next best upper-bound, select it. Else, push back to heap.

**Why**: This avoids recomputing the distance matrix $K$ times. It makes the "Selection Step" nearly instant ($<50$ms for $K=1000$).

### Step 1.3: PyO3 Bindings
*   Use `numpy` crate.
*   Ensure you use `PyReadonlyArray2` to accept data from Python without copying.
*   Zero-copy is the feature.

## Phase 2: The "Test Track" (The Task) [Days 5-7]
**Goal**: A Python function that spits out data for the 20-DOF Arm.

### Step 2.1: 20-DOF Forward Kinematics
*   Don't use a physics engine (MuJoCo/Bullet) yet. It's overkill.
*   Just write the Forward Kinematics matrix math in `jax.numpy` or standard `numpy`.
*   **Input**: 20 angles.
*   **Output**: End-effector $(x, y)$.

### Step 2.2: The "Trap"
*   To prove your algorithm works, the task must be deceptive.
*   Define a wall/obstacle in the workspace.
*   If any joint hits it $\rightarrow$ Fitness = 0.
*   This forces the algorithm to find "snaking" configurations, which requires high diversity.

## Phase 3: The "Race" (The Baseline) [Days 8-10]
**Goal**: A fair fight.

### Step 3.1: Sparse MAP-Elites
*   Since a 20D grid is impossible, you must implement MAP-Elites with a Hash Map.
*   **Logic**: `key = tuple(floor(state / cell_size))`.
*   If key exists in dict, compare fitness.

### Step 3.2: The Crash Test
*   Run Sparse MAP-Elites with 5 bins per dimension.
*   **Prediction**: It will effectively act like Random Search because almost every point will hash to a unique (empty) cell in $5^{20}$ space. This is exactly what you want to demonstrate.

## Phase 4: The "Money Shot" (Integration & Plotting) [Days 11-14]
**Goal**: The Artifact to send Jeff Clune.

### Step 4.1: The Loop
*   Connect `mmr_elites_rs.select()` to the Python evolution loop.

### Step 4.2: The Plot
*   **X-Axis**: Evaluations ($0$ to $100k$).
*   **Y-Axis**: Max Fitness AND Mean Archive Distance (Diversity).
*   **Visual**: A t-SNE projection of the final archive.
    *   **MAP-Elites**: Scattered random noise.
    *   **MMR-Elites**: A structured, continuous "snake" of solutions in the latent space.