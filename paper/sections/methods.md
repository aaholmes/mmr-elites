# Methods: Maximum Marginal Relevance Selection of Elites (MMR-Elites)

## 1. Problem Formulation

Quality-Diversity (QD) algorithms maintain an archive $\mathcal{A}$ of solutions that are both high-performing and behaviorally diverse. Each solution $x$ is characterized by its fitness $f(x) \in \mathbb{R}$ and behavior descriptor $b(x) \in \mathbb{R}^D$.

Traditional MAP-Elites discretizes the behavior space into a grid and stores the best solution per cell. However, the number of cells grows as $O(B^D)$ where $B$ is bins per dimension, making MAP-Elites infeasible for $D > 10$.

## 2. Maximum Marginal Relevance for QD

We reformulate archive maintenance as submodular maximization using Maximum Marginal Relevance (MMR), borrowed from information retrieval [Carbonell & Goldstein, 1998].

Given a pool of $N$ candidate solutions, we select $K$ archive members by iteratively choosing:

$$x^* = \arg\max_{x \in \mathcal{P} \setminus \mathcal{A}} \left[ (1 - \lambda) \cdot f(x) + \lambda \cdot d_{\min}(x, \mathcal{A}) \right]$$

where:
- $\lambda \in [0, 1]$ balances fitness exploitation vs. diversity exploration
- $d_{\min}(x, \mathcal{A}) = \min_{a \in \mathcal{A}} \|b(x) - b(a)\|_2$ is the minimum distance to any archive member

The first solution selected is always the one with highest fitness. Subsequent selections trade off fitness against distance to the current archive.

## 3. Efficient Lazy Greedy Implementation

Naively, each selection requires computing distances to all archive members for all remaining candidates: $O(K \cdot N \cdot K) = O(NK^2)$.

We accelerate selection using the lazy greedy algorithm with staleness tracking. We maintain a priority queue of candidates, and only recompute their $d_{\min}$ when they reach the top of the queue and their distance is "stale" (calculated against a smaller archive).

The algorithm achieves $O(N)$ complexity in practice for $K \ll N$.

## 4. Properties

- **Fixed Memory:** MMR-Elites maintains exactly $K$ solutions regardless of behavior space dimension.
- **Uniform Coverage:** By explicitly maximizing $d_{\min}$, MMR-Elites achieves near-uniform distribution in behavior space.
- **Submodular Guarantees:** The diversity term $d_{\min}$ is submodular, providing theoretical guarantees on approximation quality.

## 5. Implementation

The lazy greedy selector is implemented in Rust for performance, with Python bindings via PyO3. It utilizes:
- Parallel initialization using Rayon.
- Incremental distance updates.
- Zero-copy array passing with `numpy-rust`.
