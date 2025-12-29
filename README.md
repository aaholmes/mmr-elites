# MMR-Elites

**Quality-Diversity as Information Retrieval: Overcoming the Curse of Dimensionality with Maximum Marginal Relevance Selection of Elites (MMR-Elites)**

MMR-Elites reformulates QD archive maintenance as submodular maximization using Maximum Marginal Relevance (MMR) from information retrieval. This allows for maintaining a high-quality, diverse archive in high-dimensional behavior spaces where grid-based methods like MAP-Elites fail due to the curse of dimensionality.

## Installation

### Prerequisites
- Python 3.8+
- Rust (for the MMR selector backend)
- Maturin (`pip install maturin`)

### Build from source
```bash
maturin develop --release
```

## Project Structure

- `mmr_elites/`: Main Python package
    - `algorithms/`: Implementations of MMR-Elites, MAP-Elites, CVT-MAP-Elites, and Random Search.
    - `tasks/`: Benchmark tasks including N-DOF Arm, Rastrigin, and Ant.
    - `metrics/`: QD metrics including the fair `qd_score_at_budget`.
    - `utils/`: Configuration and visualization utilities.
- `src/`: Rust source for the optimized MMR selector.
- `experiments/`: Experiment runner and ablation study scripts.
- `tests/`: Unit and integration test suite.
- `paper/`: Figures and draft sections for the GECCO submission.

## Running Experiments

To run a quick benchmark comparison:
```bash
PYTHONPATH=. python experiments/run_benchmark.py --algo mmr_elites --gens 100 --dof 20
```

To run the dimensionality scaling study:
```bash
PYTHONPATH=. python experiments/dimensionality_scaling.py
```

To run the lambda ablation study:
```bash
PYTHONPATH=. python experiments/lambda_ablation.py
```

## Testing

Run the test suite with pytest:
```bash
PYTHONPATH=. pytest tests/unit
```

For the full integration tests:
```bash
PYTHONPATH=. pytest tests/integration
```

## Core Contribution: MMR Selection

The selection score balances fitness and diversity:
```
Score(x) = (1 - λ) · fitness(x) + λ · d_min(x, Archive)
```
MMR-Elites uses an optimized lazy greedy algorithm implemented in Rust to achieve $O(N)$ selection complexity in practice.