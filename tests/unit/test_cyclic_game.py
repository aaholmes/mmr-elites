"""Correctness tests for the continuous rock-paper-scissors game."""

import numpy as np

from mmr_elites.games import cyclic
from mmr_elites.games.meta import meta_nash


def _angle_genome(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def test_single_strategy_maximally_exploitable():
    """One pure direction is maximally exploitable (NashConv = 2)."""
    assert abs(cyclic.exploitability(_angle_genome(0.7)) - 2.0) < 1e-9


def test_antipodal_pair_unexploitable():
    """Two opposite directions cancel: the 50/50 mix is unexploitable."""
    mix = [(0.5, _angle_genome(0.3)), (0.5, _angle_genome(0.3 + np.pi))]
    assert cyclic.exploitability(mix) < 1e-9


def test_uniform_spread_unexploitable():
    """Many directions evenly spread around the circle approach Nash (~0)."""
    thetas = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    mix = [(1.0, _angle_genome(t)) for t in thetas]
    assert cyclic.exploitability(mix) < 1e-6


def test_clustered_mix_is_exploitable():
    """Directions packed in a narrow arc cannot cancel -> exploitable."""
    thetas = np.linspace(0, 0.4, 10)
    mix = [(1.0, _angle_genome(t)) for t in thetas]
    assert cyclic.exploitability(mix) > 1.5


def test_payoff_antisymmetric():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(8, 2))
    m = cyclic.payoff_matrix(s, s)
    assert np.allclose(m, -m.T, atol=1e-12)


def test_meta_nash_recovers_balance_from_spread_archive():
    """Given a circle-spanning archive, the meta-Nash mixture is unexploitable."""
    thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    arch = np.array([_angle_genome(t) for t in thetas])
    p = meta_nash(cyclic.payoff_matrix(arch, arch))
    mix = [(float(w), arch[i]) for i, w in enumerate(p) if w > 1e-6]
    assert cyclic.exploitability(mix) < 1e-6


def test_meta_nash_cannot_fix_clustered_archive():
    """If the archive is all in one arc, even its best mixture stays exploitable."""
    thetas = np.linspace(0, 0.5, 24)
    arch = np.array([_angle_genome(t) for t in thetas])
    p = meta_nash(cyclic.payoff_matrix(arch, arch))
    mix = [(float(w), arch[i]) for i, w in enumerate(p) if w > 1e-6]
    assert cyclic.exploitability(mix) > 1.0
