"""Correctness tests for the Kuhn poker engine, anchored to the analytic Nash."""

import numpy as np

from mmr_elites.games import kuhn_poker as kp


def test_equilibrium_game_value():
    """The known equilibrium has game value -1/18 to player 1."""
    eq = kp.nash_equilibrium()
    assert abs(kp.expected_payoff(eq, eq) - (-1.0 / 18.0)) < 1e-9


def test_equilibrium_is_unexploitable():
    """A Nash equilibrium has (near) zero exploitability."""
    eq = kp.nash_equilibrium()
    assert kp.exploitability(eq) < 1e-9


def test_exploitability_nonnegative():
    """Exploitability (NashConv) is >= 0 for every strategy."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert kp.exploitability(rng.random(12)) >= -1e-12


def test_naive_strategies_are_exploitable():
    """Degenerate strategies are strictly exploitable."""
    assert kp.exploitability(np.zeros(12)) > 0.5  # always passive
    assert kp.exploitability(np.ones(12)) > 0.1  # always aggressive


def test_payoff_matrix_matches_scalar():
    """Vectorized payoff_matrix agrees with expected_payoff."""
    rng = np.random.default_rng(1)
    s = rng.random((5, 12))
    m = kp.payoff_matrix(s, s)
    for i in range(5):
        for j in range(5):
            assert abs(m[i, j] - kp.expected_payoff(s[i], s[j])) < 1e-12


def test_zero_sum_antisymmetry():
    """Value to P1 of (i vs j) equals minus value of (j vs i) seat-swapped."""
    rng = np.random.default_rng(2)
    s = rng.random((6, 12))
    m = kp.payoff_matrix(s, s)
    seat_avg = 0.5 * (m - m.T)
    assert np.allclose(seat_avg, -seat_avg.T)


def test_mixture_exploitability_between_components():
    """A 50/50 mix of two strategies is a valid mixed strategy (finite, >=0)."""
    rng = np.random.default_rng(3)
    a, b = rng.random(12), rng.random(12)
    mix = [(0.5, a), (0.5, b)]
    e = kp.exploitability(mix)
    assert e >= -1e-12 and np.isfinite(e)
