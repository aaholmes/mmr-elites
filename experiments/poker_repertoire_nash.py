"""
Does diversity in the archive make a repertoire harder to exploit?

We evolve a population of Kuhn poker strategies under self-play (each
candidate's fitness is its seat-averaged value against the current archive,
PSRO-style), maintaining the archive three ways that differ ONLY in the
selection rule:

  - MMR-Elites: grid-free, fixed size K, explicit (1-λ)·fitness + λ·diversity
  - MAP-Elites: grid over the (bluff, value-aggression) behavior space
  - Fitness-only: top-K by fitness (no diversity term)

Then we compute the Nash mixed strategy over each final archive (a symmetric
zero-sum meta-game solved by linear programming) and measure its
exploitability in the *true* game via an exact best response. The hypothesis,
from the population-diversity view of PSRO (Lanctot 2017; Balduzzi 2019): a
behaviorally diverse support yields a less exploitable meta-strategy, so the
two diversity-preserving archives should beat fitness-only, and MMR-Elites
should match grid MAP-Elites with a fixed-size, resolution-free archive.

Exploitability is exact (no sampling); the engine is unit-tested against the
analytic Kuhn equilibrium (value -1/18, zero exploitability).

Usage:
    python experiments/poker_repertoire_nash.py [--seeds 10] [--generations 300]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from mmr_elites.games import kuhn_poker as kp

try:
    from mmr_elites import mmr_elites_rs

    RUST = True
except ImportError:
    RUST = False

K = 64  # archive size
BATCH = 64  # offspring per generation
SIGMA = 0.1  # mutation std
N_INF = kp.N_INFOSETS


def seat_avg_matrix(pool: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Seat-averaged value of each pool strategy vs each field strategy."""
    p = kp.payoff_matrix(pool, field)  # pool as P1
    q = kp.payoff_matrix(field, pool)  # field as P1
    return 0.5 * (p - q.T)


def fitness_vs_field(pool: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Each pool member's mean seat-averaged value against the field (uniform)."""
    return seat_avg_matrix(pool, field).mean(axis=1)


def nash_over_archive(archive: np.ndarray) -> np.ndarray:
    """Symmetric zero-sum Nash mixture over archive members via LP.

    Maximize v s.t. (A^T p)_j >= v for all j, sum p = 1, p >= 0, where
    A[i,j] is the seat-averaged value of member i vs member j (antisymmetric).
    """
    a = seat_avg_matrix(archive, archive)
    n = a.shape[0]
    # variables: [p_0..p_{n-1}, v]; minimize -v
    c = np.zeros(n + 1)
    c[-1] = -1.0
    # -A^T p + v <= 0  ->  for each j: v - sum_i p_i A[i,j] <= 0
    a_ub = np.hstack([-a.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    b_eq = [1.0]
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    p = np.clip(res.x[:n], 0, None)
    return p / p.sum()


def select_mmr(pool, fitness, desc, lam):
    sel = mmr_elites_rs.MMRSelector(K, lam)
    idx = sel.select(np.ascontiguousarray(fitness), np.ascontiguousarray(desc))
    return np.asarray(idx)


def select_topk(pool, fitness, desc, _lam=None):
    return np.argsort(fitness)[-K:]


def select_map_elites(pool, fitness, desc, bins=8):
    """Keep the best-fitness member per occupied grid cell of the 2-D descriptor."""
    cells = np.clip((desc * bins).astype(int), 0, bins - 1)
    best = {}
    for i, (cx, cy) in enumerate(map(tuple, cells)):
        if (cx, cy) not in best or fitness[i] > fitness[best[(cx, cy)]]:
            best[(cx, cy)] = i
    return np.array(list(best.values()))


def evolve(method, lam, generations, rng):
    """Run self-play evolution; return the final archive (array of strategies)."""
    archive = rng.random((K, N_INF))
    for _ in range(generations):
        parents = archive[rng.integers(0, len(archive), BATCH)]
        offspring = np.clip(parents + rng.normal(0, SIGMA, parents.shape), 0, 1)
        pool = np.vstack([archive, offspring])
        fitness = fitness_vs_field(pool, archive)
        desc = np.array([kp.behavior_descriptor(s) for s in pool])
        if method == "mmr":
            idx = select_mmr(pool, fitness, desc, lam)
        elif method == "map":
            idx = select_map_elites(pool, fitness, desc)
        else:
            idx = select_topk(pool, fitness, desc)
        archive = pool[idx]
    return archive


def run(n_seeds, generations, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        ("MMR-Elites (λ=0.5)", "mmr", 0.5),
        ("MAP-Elites (8×8 grid)", "map", None),
        ("Fitness-only (top-K)", "topk", None),
    ]
    results = {name: [] for name, _, _ in methods}

    for name, method, lam in methods:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            archive = evolve(method, lam, generations, rng)
            p = nash_over_archive(archive)
            mix = [(float(w), archive[i]) for i, w in enumerate(p) if w > 1e-6]
            expl = kp.exploitability(mix)
            results[name].append(
                {"seed": seed, "archive_size": len(archive), "exploitability": expl}
            )
            print(
                f"  {name:24s} seed={seed} "
                f"|support|={len(mix):3d} exploitability={expl:.4f}"
            )

    with open(output_dir / "poker_repertoire_nash.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n=== Meta-Nash exploitability (lower = harder to exploit) ===")
    print(f"(Analytic Nash equilibrium = 0.000; random single strategy ≈ 0.4-0.9)")
    for name, _, _ in methods:
        e = np.array([r["exploitability"] for r in results[name]])
        s = np.array([r["archive_size"] for r in results[name]])
        print(f"{name:24s} {e.mean():.4f} ± {e.std():.4f}   (archive {s.mean():.0f})")

    return results


if __name__ == "__main__":
    if not RUST:
        raise SystemExit("Rust backend required: run `maturin develop --release`")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("results/poker_repertoire_nash")
    )
    args = parser.parse_args()
    run(args.seeds, args.generations, args.output)
