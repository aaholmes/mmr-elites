"""
Where diversity DOES help: an intransitive game.

Same setup as experiments/poker_repertoire_nash.py, but on continuous
rock-paper-scissors (mmr_elites.games.cyclic) instead of Kuhn poker. The game
is fully cyclic, so the Nash equilibrium is a genuinely spread-out mixture and
a clustered repertoire is exploitable no matter how you mix it.

Under self-play, fitness-only selection chases the current best response and the
population *cycles* -- it stays clustered, so its meta-Nash mixture stays
exploitable. Diversity-preserving selection (MMR-Elites or MAP-Elites) keeps the
archive spread around the circle, and the meta-Nash over a spread archive
cancels out to near-zero exploitability. This is the regime the Kuhn poker
experiment lacked: here behavioral diversity *is* strategic diversity.

Usage:
    python experiments/cyclic_game_diversity.py [--seeds 10] [--generations 300]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mmr_elites.games import cyclic
from mmr_elites.games.meta import meta_nash

try:
    from mmr_elites import mmr_elites_rs

    RUST = True
except ImportError:
    RUST = False

K = 64  # archive size
BATCH = 64  # offspring per generation


def fitness_vs_field(pool, field):
    return cyclic.payoff_matrix(pool, field).mean(axis=1)


def descriptors(pool):
    return np.array([cyclic.behavior_descriptor(s) for s in pool])


def select_mmr(fitness, desc, lam):
    sel = mmr_elites_rs.MMRSelector(K, lam)
    return np.asarray(
        sel.select(np.ascontiguousarray(fitness), np.ascontiguousarray(desc))
    )


def select_topk(fitness, desc):
    return np.argsort(fitness)[-K:]


def select_map(fitness, desc, bins=8):
    cells = np.clip((desc * bins).astype(int), 0, bins - 1)
    best = {}
    for i, c in enumerate(map(tuple, cells)):
        if c not in best or fitness[i] > fitness[best[c]]:
            best[c] = i
    return np.array(list(best.values()))


def evolve(method, lam, generations, rng):
    archive = cyclic.random_genomes(rng, K)
    for _ in range(generations):
        parents = archive[rng.integers(0, len(archive), BATCH)]
        offspring = cyclic.mutate(parents, rng)
        pool = np.vstack([archive, offspring])
        fitness = fitness_vs_field(pool, archive)
        desc = descriptors(pool)
        if method == "mmr":
            idx = select_mmr(fitness, desc, lam)
        elif method == "map":
            idx = select_map(fitness, desc)
        else:
            idx = select_topk(fitness, desc)
        archive = pool[idx]
    return archive


def archive_spread(archive):
    """Mean resultant length of the archive's own directions (1 = clustered)."""
    t = np.arctan2(archive[:, 1], archive[:, 0])
    return float(np.abs(np.mean(np.exp(1j * t))))


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
            p = meta_nash(cyclic.payoff_matrix(archive, archive))
            mix = [(float(w), archive[i]) for i, w in enumerate(p) if w > 1e-6]
            expl = cyclic.exploitability(mix)
            results[name].append(
                {
                    "seed": seed,
                    "exploitability": expl,
                    "archive_clustering": archive_spread(archive),
                    "support": len(mix),
                }
            )
            print(
                f"  {name:24s} seed={seed} exploitability={expl:.4f} "
                f"clustering={archive_spread(archive):.3f}"
            )

    with open(output_dir / "cyclic_game_diversity.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n=== Meta-Nash exploitability on continuous RPS (lower = better) ===")
    print("(Nash equilibrium = 0; a single pure strategy = 2.0)")
    for name, _, _ in methods:
        e = np.array([r["exploitability"] for r in results[name]])
        c = np.array([r["archive_clustering"] for r in results[name]])
        print(
            f"{name:24s} exploitability {e.mean():.4f} ± {e.std():.4f}   "
            f"archive clustering {c.mean():.3f}"
        )

    return results


if __name__ == "__main__":
    if not RUST:
        raise SystemExit("Rust backend required: run `maturin develop --release`")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("results/cyclic_game_diversity")
    )
    args = parser.parse_args()
    run(args.seeds, args.generations, args.output)
