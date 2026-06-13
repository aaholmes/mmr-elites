"""Tests for the skill + style (transitive + cyclic) game."""

import numpy as np

from mmr_elites.games import skill_cyclic as sc


def _g(skill_logit, theta):
    return np.array([skill_logit, theta])


def test_closed_form_matches_bruteforce():
    """Closed-form exploitability matches grid-search best response."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = rng.integers(1, 8)
        strats = rng.normal(size=(n, 2))
        w = rng.random(n)
        mix = [(float(wk), s) for wk, s in zip(w, strats)]
        assert abs(sc.exploitability(mix) - sc.exploitability_bruteforce(mix)) < 1e-3


def test_payoff_antisymmetric():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(8, 2))
    m = sc.payoff_matrix(s, s)
    assert np.allclose(m, -m.T, atol=1e-12)


def test_needs_both_skill_and_balance():
    thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    high_balanced = [(1.0, _g(8.0, t)) for t in thetas]
    high_clustered = [(1.0, _g(8.0, t)) for t in np.linspace(0, 0.3, 24)]
    low_balanced = [(1.0, _g(-8.0, t)) for t in thetas]

    # Only high-skill AND balanced is unexploitable.
    assert sc.exploitability(high_balanced) < 0.05
    # Clustered style is exploitable cyclically even at max skill.
    assert sc.exploitability(high_clustered) > 1.0
    # Balanced style is exploitable transitively if skill is low.
    assert sc.exploitability(low_balanced) > 1.5


def test_single_strategy_exploitable():
    assert sc.exploitability(_g(8.0, 0.3)) > 1.0  # max skill, single style
