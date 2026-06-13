"""
Skill + style: a zero-sum game with BOTH a transitive and a cyclic axis.

Every strategy has a skill r in (0, 1) and a style angle theta. The payoff to x
against y is

    u(x, y) = (r_x - r_y)  +  kappa * r_x * r_y * sin(theta_x - theta_y)

The first term is purely transitive -- higher skill beats lower skill regardless
of style (a genuine quality dimension). The second is purely cyclic -- a
rock-paper-scissors over styles, amplified by skill (better players have sharper
matchup edges and exposures). The game is zero-sum (u is antisymmetric).

This is the realistic regime the pure cyclic game lacked: being good requires
*both* high skill and unpredictable style. Best-responding to any mixture, a
deviator sets its own skill to the max and aligns its style to punish the
population's net style bias, which gives a closed form for exploitability:

    exploitability = 2 * (1 - mean_skill)  +  2 * kappa * | skill-weighted style resultant |
                   = transitive gap        +  cyclic gap

It is zero only when the repertoire is simultaneously max-skill (mean_skill = 1)
and style-balanced (the skill-weighted directions cancel). Pure fitness drives
skill up but clusters style (cyclic gap large); pure diversity spreads style but
abandons skill (transitive gap large). Only a high-skill, style-diverse
repertoire is unexploitable -- exactly the balance MMR's lambda controls.

kappa < 1 keeps the deviator's best skill pinned at the maximum (so the closed
form holds); we use 0.8.
"""

from typing import List, Tuple

import numpy as np

GENOME_DIM = 2
KAPPA = 0.8


def skill(genome: np.ndarray) -> np.ndarray:
    """Skill r in (0, 1), a saturating function of the first genome coordinate."""
    g = np.atleast_2d(genome)
    return 1.0 / (1.0 + np.exp(-g[:, 0]))


def _theta(genome: np.ndarray) -> np.ndarray:
    return np.atleast_2d(genome)[:, 1]


def payoff_matrix(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    r1 = skill(s1)[:, None]
    r2 = skill(s2)[None, :]
    t1 = _theta(s1)[:, None]
    t2 = _theta(s2)[None, :]
    return (r1 - r2) + KAPPA * r1 * r2 * np.sin(t1 - t2)


def behavior_descriptor(genome: np.ndarray) -> np.ndarray:
    """Style only (skill is the quality axis, left to fitness, not diversity)."""
    t = _theta(genome)[0]
    return np.array([(np.cos(t) + 1) / 2, (np.sin(t) + 1) / 2])


def _items(strat_or_mix):
    if isinstance(strat_or_mix, np.ndarray):
        return [(1.0, strat_or_mix)]
    return list(strat_or_mix)


def exploitability(strat_or_mix) -> float:
    """Closed-form NashConv: transitive gap + cyclic gap (both >= 0)."""
    items = _items(strat_or_mix)
    w = np.array([wk for wk, _ in items], dtype=float)
    w = w / w.sum()
    r = np.array([skill(g)[0] for _, g in items])
    th = np.array([_theta(g)[0] for _, g in items])
    mean_skill = float(np.sum(w * r))
    resultant = np.abs(np.sum(w * r * np.exp(1j * th)))
    return 2.0 * (1.0 - mean_skill) + 2.0 * KAPPA * float(resultant)


def exploitability_bruteforce(strat_or_mix, grid: int = 720) -> float:
    """Independent check: best-response by grid search over (skill, style)."""
    items = _items(strat_or_mix)
    w = np.array([wk for wk, _ in items], dtype=float)
    w = w / w.sum()
    r = np.array([skill(g)[0] for _, g in items])
    th = np.array([_theta(g)[0] for _, g in items])
    rbar = np.sum(w * r)
    res = np.sum(w * r * np.exp(1j * th))  # skill-weighted resultant
    rc = np.linspace(0.0, 1.0, 50)[:, None]
    phi = np.linspace(0, 2 * np.pi, grid, endpoint=False)[None, :]
    # value to a P1 deviator (rc, phi) vs the mixture
    m_phi = np.imag(np.exp(1j * phi) * np.conj(res))  # E[r_k sin(phi - th_k)]
    v1 = (rc - rbar) + KAPPA * rc * m_phi
    b1 = v1.max()
    # value to P1 when a P2 deviator (rc, phi) best-responds (minimizes)
    v2 = (rbar - rc) - KAPPA * rc * m_phi
    b2 = v2.min()
    return float(b1 - b2)


def random_genomes(rng, n: int) -> np.ndarray:
    return rng.normal(size=(n, GENOME_DIM))


def mutate(parents: np.ndarray, rng, sigma: float = 0.3) -> np.ndarray:
    return parents + rng.normal(0, sigma, parents.shape)
