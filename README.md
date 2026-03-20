# MMR-Elites

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In many settings one wants to choose a small subset from a large pool of candidate solutions that is not only high-quality but also diverse. Selecting by quality alone produces redundancy: the best items tend to cluster together.

This repo implements an efficient algorithm that solves this by balancing quality with diversity, using an old idea from information retrieval called Maximal Marginal Relevance (MMR).

We call this algorithm MMR-Elites, in homage to both MMR and [MAP-Elites](https://arxiv.org/abs/1504.04909), a popular quality-diversity algorithm.

## 🎯 Example: LLM Response Selection

As an example, consider selecting the 10 best responses from 50 LLM-generated pieces of advice about startup fundraising, where quality is scored by another LLM and semantic similarity is measured with an embedding model.

Naive top-K grabs the highest-scoring responses, but they cluster around similar themes. MMR-Elites selects responses that are both high-quality *and* semantically distinct:

| Method | Top-1 Quality | Mean Quality | Cosine Diversity |
|--------|:------------:|:-----------:|:---------------:|
| Naive Top-K | 1.000 | 0.620 | 0.653 |
| **MMR-Elites** | **1.000** | **0.608** | **0.716** |

Top-1 quality is always identical: MMR's greedy selection guarantees the best item is picked first. Subsequent picks balance quality with diversity from already-selected items, trading 2% mean quality for **10% higher diversity**. In practice, this means swapping redundant responses (e.g., a second "investor research" tip) for semantically distinct ones ("competitive rounds", "warm intros").

```bash
# Try it yourself (pre-generated responses included, no API key needed)
pip install -e ".[examples]"
python examples/llm_response_selection.py
```

## 🔬 How It Works

Traditional Quality-Diversity (QD) algorithms like MAP-Elites discretize behavior space into grids, which scales exponentially with dimension (3²⁰ = 3.5 billion cells for a 20-DOF arm). MMR-Elites reformulates archive maintenance as submodular maximization, enabling:
- **O(K) fixed memory** regardless of behavior space dimension
- **Uniform coverage** via explicit diversity optimization
- **O(K log K)** selection via lazy greedy algorithm
- **Scalable to high-D** behavior spaces where MAP-Elites fails

### The MMR Selection Criterion

At each generation, we select K survivors from the pool of archive + offspring by iteratively choosing the solution that maximizes:

```
x* = argmax[(1 - λ) · fitness(x) + λ · d_min(x, Selected)]
```

Where:
- **λ = 0**: Pure fitness selection (top-K by fitness)
- **λ = 1**: Pure diversity selection (maximize spread)
- **λ = 0.5**: Balanced selection (recommended default)

### Saturating Distance Functions

In practice we don't really want to *maximize* diversity, but just make sure that the solutions are *different enough* from each other. So, we avoid using simple Euclidean distances between semantic embeddings as they grow unbounded especially in high-dimensional behavior spaces, making the diversity term dominate. Instead, we use **exponential saturation** to bound distances to [0, 1]:

```
d_sat(b₁, b₂) = 1 - exp(-||b₁ - b₂|| / σ)
```

This has a key advantage over Gaussian saturation `1 - exp(-||b₁-b₂||²/2σ²)`: it maintains a **linear gradient at small distances**, avoiding the "dead zone" where Gaussian saturation returns near-zero values for nearby solutions. This means the selector can still discriminate between close neighbors — critical for maintaining fine-grained diversity in dense regions of the archive.

### Efficient Lazy Greedy Algorithm

Naive selection is O(NK²). We achieve O(K log K) in practice using:

1. **Staleness tracking**: Cache `d_min` and only recompute when the archive changes
2. **Priority queue**: Candidates sorted by upper-bound scores
3. **Early termination**: Accept candidate if current score beats all upper bounds

The Rust implementation achieves ~50x speedup over pure Python.

## 📊 QD Benchmark Results

MMR-Elites achieves **12x better uniformity** than MAP-Elites and **6x better than CVT-MAP-Elites** on a 20-DOF arm reaching task:

| Algorithm | QD-Score@K | Uniformity (CV↓) | Archive Size |
|-----------|-----------|------------------|--------------|
| **MMR-Elites** | 552.3 ± 0.2 | **0.066** | 1,000 |
| CVT-MAP-Elites | 497.6 ± 0.9 | 0.417 | 758 |
| MAP-Elites | 7468.6 ± 34.1 | 0.832 | 22,825 |
| Random | 583.0 ± 1.2 | 0.789 | 1,000 |

*Lower uniformity CV = more uniform coverage. MAP-Elites achieves higher raw QD-Score due to unbounded archive, but with poor uniformity.*

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aaholmes/mmr-elites.git
cd mmr-elites

# Install Rust backend (required)
pip install maturin
maturin develop --release

# Install Python package
pip install -e .
```

### LLM Response Selection

See the results table above. To regenerate responses with your own Gemini API key:

```bash
pip install -e ".[examples]"
GEMINI_API_KEY=... python examples/generate_responses.py
python examples/llm_response_selection.py
```

### QD Benchmarks

```bash
# Run a quick experiment
mmr-elites run --task arm --generations 500 --seed 42

# Compare all algorithms
mmr-elites benchmark --quick

# Dimensionality scaling study
mmr-elites compare --dimensions 5 10 20 50 100

# Launch interactive demo
mmr-elites demo
```

### Python API

```python
from mmr_elites.tasks.arm import ArmTask
from mmr_elites.algorithms import run_mmr_elites

# Create task
task = ArmTask(n_dof=20, use_highdim_descriptor=True)

# Run MMR-Elites
result = run_mmr_elites(
    task,
    archive_size=1000,
    generations=2000,
    lambda_val=0.5,  # Balance fitness and diversity
    seed=42
)

print(f"QD-Score: {result.final_metrics['qd_score']:.2f}")
print(f"Max Fitness: {result.final_metrics['max_fitness']:.4f}")
print(f"Uniformity: {result.final_metrics['uniformity_cv']:.4f}")
```

## 📁 Project Structure

```
mmr-elites/
├── mmr_elites/           # Main package
│   ├── algorithms/       # MMR-Elites, MAP-Elites, CVT-MAP-Elites
│   ├── tasks/           # Benchmark tasks (Arm, Rastrigin)
│   ├── metrics/         # QD metrics
│   └── utils/           # Config, visualization, statistics
├── src/lib.rs           # Rust MMR selector
├── examples/           # Standalone demo (LLM response selection)
├── experiments/         # Experiment scripts
├── tests/              # Test suite
└── demo/               # Interactive Streamlit demo
```

## 📖 Citation

If you use this code, please cite:

```bibtex
@misc{holmes2026mmrelites,
  title={Quality-Diversity as Information Retrieval:
         Overcoming the Curse of Dimensionality with
         Maximum Marginal Relevance Selection of Elites},
  author={Holmes, Adam A.},
  year={2026},
  note={arXiv preprint forthcoming}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- The MMR formulation is inspired by Carbonell & Goldstein (1998)
- MAP-Elites was introduced by Mouret & Clune (2015); reading about it in Risi et al.'s [*Neuroevolution*](https://neuroevolutionbook.com/) (MIT Press, 2025) is what led to thinking about alternatives that overcome the curse of dimensionality
- Benchmark tasks adapted from the pyribs library
- Submodular optimization insights from Krause & Golovin (2014)
