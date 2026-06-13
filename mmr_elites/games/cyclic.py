"""
Continuous rock-paper-scissors: a fully intransitive zero-sum game.

A strategy is a direction on a circle (genome = a 2-D vector, its angle is what
matters). The payoff is

    u(x, y) = sin(theta_x - theta_y)

so you beat strategies "just behind" you and lose to those "just ahead" --
a smooth, fully cyclic generalization of rock-paper-scissors with no transitive
component at all. This is the canonical setting (cf. Balduzzi et al. 2019,
"Open-ended learning in symmetric zero-sum games") where a *single* strategy is
maximally exploitable and the only safe play is a mixture whose directions
cancel out. The Nash equilibrium is the uniform distribution over the circle.

Exploitability has a closed form. Against a mixture with weights w_k at angles
theta_k, the best response picks the angle maximizing the expected payoff, whose
value is the length of the mean resultant vector R = sum_k w_k e^{i theta_k}.
Both seats are symmetric, so

    exploitability = 2 * |R|.

A clustered repertoire has |R| ~ 1 (exploitable); a repertoire whose directions
spread around the circle has |R| ~ 0 (unexploitable). Behavioral diversity and
strategic diversity coincide here -- the opposite of Kuhn poker.
"""

from typing import List, Sequence, Tuple

import numpy as np

GENOME_DIM = 2


def _angles(genomes: np.ndarray) -> np.ndarray:
    g = np.atleast_2d(genomes)
    return np.arctan2(g[:, 1], g[:, 0])


def payoff_matrix(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Value-to-row matrix M[i, j] = sin(theta_i - theta_j)."""
    t1 = _angles(s1)[:, None]
    t2 = _angles(s2)[None, :]
    return np.sin(t1 - t2)


def behavior_descriptor(genome: np.ndarray) -> np.ndarray:
    """2-D descriptor: the unit direction mapped to [0, 1]^2."""
    t = _angles(genome)[0]
    return np.array([(np.cos(t) + 1) / 2, (np.sin(t) + 1) / 2])


def exploitability(strat_or_mix) -> float:
    """NashConv via the mean resultant vector; 0 = unexploitable."""
    if isinstance(strat_or_mix, np.ndarray):
        items: List[Tuple[float, np.ndarray]] = [(1.0, strat_or_mix)]
    else:
        items = list(strat_or_mix)
    angles = np.array([_angles(g)[0] for _, g in items])
    w = np.array([wk for wk, _ in items], dtype=float)
    w = w / w.sum()
    resultant = np.sum(w * np.exp(1j * angles))
    return float(2.0 * np.abs(resultant))


def random_genomes(rng, n: int) -> np.ndarray:
    return rng.normal(size=(n, GENOME_DIM))


def mutate(parents: np.ndarray, rng, sigma: float = 0.3) -> np.ndarray:
    return parents + rng.normal(0, sigma, parents.shape)
