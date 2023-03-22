"""Interface for two-player zero-sum normal-form games.
"""
from typing import Optional

import numpy as np


class Game:
    def __init__(self, payoff_table: np.ndarray):
        """Two-player zero-sum normal-form game

        Args:
            payoff_table: A 2-d array containing payoffs for joint actions
                Player 1's payoffs are equal to the entries, player 2's payoffs
                are equal to the negations of the entries

        Attributes:
            payoff_table: Array containing the payoffs for joint actions
            n_actions: The number of actions for each player
        """
        self.payoff_table: np.ndarray = payoff_table
        self.n_actions: tuple = self.payoff_table.shape

    def compute_payoffs(
        self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute action values for one player, given the other player's policy

        Args:
            a: Policy for player one
            b: Policy for player two
        Exactly one of `a` and `b` should be passed as input
        """
        if a is not None and b is None:
            return -np.dot(a, self.payoff_table)
        if a is None and b is not None:
            return np.dot(self.payoff_table, b)
        raise ValueError

    def exploitability(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Compute exploitability

        Args:
            a: Policy for player one
            b: Policy for player two
        At least one of `a` and `b` should be passed as input
        """
        if a is None:
            return self.compute_payoffs(None, b).max()
        if b is None:
            return self.compute_payoffs(a, None).max()
        return (self.exploitability(a, None) + self.exploitability(None, b)) / 2
