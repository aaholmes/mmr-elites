"""
Kuhn poker: an exactly-solvable two-player zero-sum game.

Kuhn poker is the smallest non-trivial imperfect-information poker game (3
cards J<Q<K, one betting round). Everything here is computed by exact game-tree
traversal -- no sampling -- so payoffs and exploitability are deterministic and
checkable against the known analytic equilibrium (game value -1/18 to player 1,
zero exploitability). That makes it a trustworthy substrate for studying what a
*repertoire* of evolved strategies buys you game-theoretically.

Strategy representation
-----------------------
A behavioral strategy is a length-12 vector of probabilities, one per
information set, giving the probability of the "aggressive" action:

    index = card * 4 + context
    card:    0=J, 1=Q, 2=K
    context: 0 = P1 opening      -> aggressive = bet,  passive = check
             1 = P1 facing a bet -> aggressive = call, passive = fold
             2 = P2 facing check  -> aggressive = bet,  passive = check
             3 = P2 facing a bet  -> aggressive = call, passive = fold

Payoffs are reported as expected value to player 1 (zero-sum). Antes are 1 each;
a called bet plays for +/-2, a fold loses the 1 ante, a check-check showdown is
+/-1.

Best response to a *mixture* of strategies (a repertoire deployed as a mixed
strategy) is computed by accumulating counterfactual reach over the mixture
components and picking the best action per information set -- the textbook
best-response-to-a-mixture, exact for any population.
"""

from typing import List, Sequence, Tuple

import numpy as np

N_INFOSETS = 12


def _idx(card: int, context: int) -> int:
    return card * 4 + context


def expected_payoff(s1: np.ndarray, s2: np.ndarray) -> float:
    """Expected value to player 1 when s1 plays P1 and s2 plays P2."""
    total = 0.0
    for a in range(3):  # P1 card
        for b in range(3):  # P2 card
            if a == b:
                continue
            show = 1.0 if a > b else -1.0
            p1_bet = s1[_idx(a, 0)]
            p1_call = s1[_idx(a, 1)]
            p2_bet = s2[_idx(b, 2)]
            p2_call = s2[_idx(b, 3)]

            # P1 bets -> P2 calls (showdown for 2) or folds (P1 wins 1)
            val_bet = p2_call * (2 * show) + (1 - p2_call) * 1.0
            # P1 checks -> P2 bets or checks
            val_check_p2bet = p1_call * (2 * show) + (1 - p1_call) * (-1.0)
            val_check = (1 - p2_bet) * show + p2_bet * val_check_p2bet

            v = p1_bet * val_bet + (1 - p1_bet) * val_check
            total += v / 6.0
    return total


def payoff_matrix(s1_batch: np.ndarray, s2_batch: np.ndarray) -> np.ndarray:
    """Value-to-P1 matrix M[i, j] for s1_batch[i] as P1 vs s2_batch[j] as P2.

    Vectorized equivalent of expected_payoff over all pairs.
    """
    s1 = np.atleast_2d(s1_batch)
    s2 = np.atleast_2d(s2_batch)
    m = np.zeros((s1.shape[0], s2.shape[0]))
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            show = 1.0 if a > b else -1.0
            p1_bet = s1[:, _idx(a, 0)][:, None]
            p1_call = s1[:, _idx(a, 1)][:, None]
            p2_bet = s2[:, _idx(b, 2)][None, :]
            p2_call = s2[:, _idx(b, 3)][None, :]
            val_bet = p2_call * (2 * show) + (1 - p2_call) * 1.0
            val_check_p2bet = p1_call * (2 * show) + (1 - p1_call) * (-1.0)
            val_check = (1 - p2_bet) * show + p2_bet * val_check_p2bet
            m += (p1_bet * val_bet + (1 - p1_bet) * val_check) / 6.0
    return m


Mixture = Sequence[Tuple[float, np.ndarray]]


def _as_mixture(strat_or_mix) -> List[Tuple[float, np.ndarray]]:
    if isinstance(strat_or_mix, np.ndarray):
        return [(1.0, strat_or_mix)]
    return list(strat_or_mix)


def best_response_value_p1(opponent) -> float:
    """Max value to P1 when P1 best-responds to a P2 strategy/mixture."""
    mix = _as_mixture(opponent)
    b1 = 0.0
    for a in range(3):  # P1 card
        # Decision at P1-facing-a-bet (call vs fold), counterfactual over b/members.
        call_sum = fold_sum = 0.0
        for w, s2 in mix:
            for b in range(3):
                if b == a:
                    continue
                q = s2[_idx(b, 2)]  # P2 bets after a check
                cw = w * q / 6.0
                show = 1.0 if a > b else -1.0
                call_sum += cw * (2 * show)
                fold_sum += cw * (-1.0)
        checkbet_value = max(call_sum, fold_sum)

        # Opening decision (bet vs check); P1 maximizes value.
        bet_total = check_show = 0.0
        for w, s2 in mix:
            for b in range(3):
                if b == a:
                    continue
                show = 1.0 if a > b else -1.0
                p2_call = s2[_idx(b, 3)]
                q = s2[_idx(b, 2)]
                bet_total += w / 6.0 * (p2_call * (2 * show) + (1 - p2_call) * 1.0)
                check_show += w / 6.0 * ((1 - q) * show)
        b1 += max(bet_total, check_show + checkbet_value)
    return b1


def best_response_value_p2(opponent) -> float:
    """Min value to P1 when P2 best-responds to a P1 strategy/mixture.

    (P2 minimizes value-to-P1 because the game is zero-sum.)
    """
    mix = _as_mixture(opponent)
    b2 = 0.0
    for b in range(3):  # P2 card
        # Facing a check: P2 checks (showdown) or bets (then fixed P1 call/fold).
        check_sum = bet_sum = 0.0
        # Facing a bet: P2 calls (showdown for 2) or folds (P1 wins 1).
        call_sum = fold_sum = 0.0
        for w, s1 in mix:
            for a in range(3):
                if a == b:
                    continue
                show = 1.0 if a > b else -1.0
                p1_bet = s1[_idx(a, 0)]
                p1_call = s1[_idx(a, 1)]

                cw_check = w * (1 - p1_bet) / 6.0
                check_sum += cw_check * show
                bet_sum += cw_check * (p1_call * (2 * show) + (1 - p1_call) * (-1.0))

                cw_bet = w * p1_bet / 6.0
                call_sum += cw_bet * (2 * show)
                fold_sum += cw_bet * 1.0
        b2 += min(check_sum, bet_sum) + min(call_sum, fold_sum)
    return b2


def exploitability(strat_or_mix) -> float:
    """NashConv: how much best responders beat the strategy in both seats.

    Zero at a Nash equilibrium; larger means more exploitable. Works for a
    single strategy or a mixture (a deployed repertoire).
    """
    return best_response_value_p1(strat_or_mix) - best_response_value_p2(strat_or_mix)


def behavior_descriptor(strat: np.ndarray) -> np.ndarray:
    """2-D behavior descriptor: (bluff rate, value-aggression rate).

    bluff   = aggressive actions with the worst card J (betting/calling light)
    value   = aggressive actions with the best card K
    Both in [0, 1]; this is the strategic-style axis a repertoire spreads over.
    """
    bluff = (strat[_idx(0, 0)] + strat[_idx(0, 1)] + strat[_idx(0, 2)]) / 3.0
    value = (strat[_idx(2, 0)] + strat[_idx(2, 1)] + strat[_idx(2, 3)]) / 3.0
    return np.array([bluff, value])


# Analytic Nash equilibrium (the alpha = 0 member of the equilibrium family),
# used to anchor correctness. Game value to P1 is -1/18.
def nash_equilibrium() -> np.ndarray:
    s = np.zeros(N_INFOSETS)
    # J: never bet, never call; bluff-bet as P2 after a check w.p. 1/3
    s[_idx(0, 2)] = 1.0 / 3.0
    # Q: never bet open; call a bet w.p. 1/3 (both seats)
    s[_idx(1, 1)] = 1.0 / 3.0
    s[_idx(1, 3)] = 1.0 / 3.0
    # K: never bet open (alpha=0); always call/bet otherwise
    s[_idx(2, 1)] = 1.0
    s[_idx(2, 2)] = 1.0
    s[_idx(2, 3)] = 1.0
    return s
