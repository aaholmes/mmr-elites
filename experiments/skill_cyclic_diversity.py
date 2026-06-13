"""
Where balancing quality AND diversity wins: the skill + style game.

On mmr_elites.games.skill_cyclic, a repertoire is unexploitable only if it is
*both* high-skill (transitive) and style-balanced (cyclic). We evolve under
self-play and maintain the archive several ways, then compute the meta-Nash
mixture over each final archive and measure its exploitability, broken into:

  - transitive gap = 2 * (1 - mean skill)              [pay for low skill]
  - cyclic gap     = 2 * kappa * |style resultant|     [pay for style bias]

Prediction: fitness-only selection maximizes skill but clusters style (cyclic
gap large); pure-diversity selection (MMR λ=1) balances style but abandons skill
(transitive gap large); an intermediate λ does both and is least exploitable --
a U-shaped curve in λ that argues for the *balance*, not diversity alone.

Usage:
    python experiments/skill_cyclic_diversity.py [--seeds 10] [--generations 300]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mmr_elites.games import skill_cyclic as sc
from mmr_elites.games.meta import meta_nash

try:
    from mmr_elites import mmr_elites_rs

    RUST = True
except ImportError:
    RUST = False

K = 64
BATCH = 64


def evolve(method, lam, generations, rng):
    archive = sc.random_genomes(rng, K)
    for _ in range(generations):
        parents = archive[rng.integers(0, len(archive), BATCH)]
        offspring = sc.mutate(parents, rng)
        pool = np.vstack([archive, offspring])
        fitness = sc.payoff_matrix(pool, archive).mean(axis=1)
        desc = np.array([sc.behavior_descriptor(s) for s in pool])
        if method == "mmr":
            sel = mmr_elites_rs.MMRSelector(K, lam)
            idx = np.asarray(
                sel.select(np.ascontiguousarray(fitness), np.ascontiguousarray(desc))
            )
        elif method == "map":
            cells = np.clip((desc * 8).astype(int), 0, 7)
            best = {}
            for i, c in enumerate(map(tuple, cells)):
                if c not in best or fitness[i] > fitness[best[c]]:
                    best[c] = i
            idx = np.array(list(best.values()))
        else:  # topk
            idx = np.argsort(fitness)[-K:]
        archive = pool[idx]
    return archive


def decompose(archive):
    """Meta-Nash mixture -> (exploitability, transitive gap, cyclic gap, mean skill)."""
    p = meta_nash(sc.payoff_matrix(archive, archive))
    keep = p > 1e-6
    w = p[keep] / p[keep].sum()
    members = archive[keep]
    r = sc.skill(members)
    th = members[:, 1]
    mean_skill = float(np.sum(w * r))
    rho = float(np.abs(np.sum(w * r * np.exp(1j * th))))
    transitive = 2.0 * (1.0 - mean_skill)
    cyclic = 2.0 * sc.KAPPA * rho
    return transitive + cyclic, transitive, cyclic, mean_skill


def run(n_seeds, generations, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        ("MMR-Elites λ=0.0", "mmr", 0.0),
        ("MMR-Elites λ=0.25", "mmr", 0.25),
        ("MMR-Elites λ=0.5", "mmr", 0.5),
        ("MMR-Elites λ=0.75", "mmr", 0.75),
        ("MMR-Elites λ=1.0", "mmr", 1.0),
        ("MAP-Elites (8×8)", "map", None),
        ("Fitness-only (top-K)", "topk", None),
    ]
    results = {name: [] for name, _, _ in methods}

    for name, method, lam in methods:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            archive = evolve(method, lam, generations, rng)
            expl, trans, cyc, skill = decompose(archive)
            results[name].append(
                {
                    "seed": seed,
                    "exploitability": expl,
                    "transitive_gap": trans,
                    "cyclic_gap": cyc,
                    "mean_skill": skill,
                }
            )
        e = np.array([r["exploitability"] for r in results[name]])
        print(f"  {name:22s} exploitability {e.mean():.3f} ± {e.std():.3f}")

    with open(output_dir / "skill_cyclic_diversity.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n=== Skill + style game: meta-Nash exploitability (lower = better) ===")
    print(
        f"{'method':22s} {'exploit':>8s} {'=transitive':>12s} {'+cyclic':>9s}  {'skill':>6s}"
    )
    for name, _, _ in methods:
        rs = results[name]
        e = np.mean([r["exploitability"] for r in rs])
        t = np.mean([r["transitive_gap"] for r in rs])
        c = np.mean([r["cyclic_gap"] for r in rs])
        s = np.mean([r["mean_skill"] for r in rs])
        print(f"{name:22s} {e:8.3f} {t:12.3f} {c:9.3f}  {s:6.3f}")

    return results


if __name__ == "__main__":
    if not RUST:
        raise SystemExit("Rust backend required: run `maturin develop --release`")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("results/skill_cyclic_diversity")
    )
    args = parser.parse_args()
    run(args.seeds, args.generations, args.output)
