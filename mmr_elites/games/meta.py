"""Meta-game solver: Nash mixture over a finite set of strategies."""

import numpy as np
from scipy.optimize import linprog


def meta_nash(payoff: np.ndarray) -> np.ndarray:
    """Nash mixture of the row player in a symmetric zero-sum meta-game.

    `payoff[i, j]` is the (antisymmetric) value to strategy i against strategy j.
    Returns a probability vector p maximizing the worst-case value
    min_j (p^T payoff)_j, i.e. the maximin / Nash mixture.
    """
    n = payoff.shape[0]
    # variables [p_0..p_{n-1}, v]; minimize -v
    c = np.zeros(n + 1)
    c[-1] = -1.0
    # for each j:  v - sum_i p_i payoff[i, j] <= 0
    a_ub = np.hstack([-payoff.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=[(0, None)] * n + [(None, None)],
    )
    p = np.clip(res.x[:n], 0, None)
    return p / p.sum()
