"""Helper class for magnetic mirror descent to do tree traversals.
"""

from enum import Enum

import numpy as np
from open_spiel.python.policy import TabularPolicy
from pyspiel import State


class Objective(Enum):
    standard = "standard"
    maxent = "maxent"
    minimaxent = "minimaxent"


class Node:
    def __init__(
        self,
        history: State,
        policies: TabularPolicy,
        reach_contributions: dict[int, float],
        temp: float,
        objective: Objective,
    ):
        """Tree traversal class for magnetic mirror descent

        Args:
            history: Game history to which node corresponds
            policies: Policies of players for traversal
            reach_contributions: Reach probability of players & chance to `history`
            temp: Regularization temperature for magnetic mirror descent
            objective: Objective for updates

        Attributes:
            history
            policies
            reach_contributions
            temp
            objective
            children: Map from actions to child nodes
            action_values: Map from actions to expected values for each player
            history_values: Expected value for each player at node
        """
        self.history = history.clone()
        self.policies = policies
        self.reach_contributions = reach_contributions
        self.temp = temp
        self.objective = objective
        self.children: dict[int, Node] = {}
        self.action_values: dict[int, np.ndarray] = {}
        self.history_values: np.ndarray
        self.__recurse()

    def __recurse(self) -> None:
        """Expand tree downward"""
        if self.history.is_terminal():
            self.history_values = np.array(self.history.returns())
            return
        if self.history.is_chance_node():
            outcomes = self.history.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
        else:
            outcome_dict = self.policies.action_probabilities(self.history)
            action_list = list(outcome_dict.keys())
            prob_list = list(outcome_dict.values())
        self.history_values = np.zeros(self.history.num_players())
        cur_player = self.history.current_player()
        for a, p in zip(action_list, prob_list):
            if p > 0:
                history_ = self.history.clone()
                history_.apply_action(a)
                reach_contributions_ = self.reach_contributions.copy()
                reach_contributions_[cur_player] *= p
                child = Node(
                    history_,
                    self.policies,
                    reach_contributions_,
                    self.temp,
                    self.objective,
                )
                self.children[a] = child
                child_values = child.history_values
                if cur_player == -1:  # chance
                    ent_reward = 0
                else:
                    if self.objective == Objective.standard:
                        coefs = np.zeros(self.history.num_players())
                    elif self.objective == Objective.maxent:
                        coefs = np.zeros(self.history.num_players())
                        coefs[cur_player] = 1
                    elif self.objective == Objective.minimaxent:
                        coefs = -np.ones(self.history.num_players())
                        coefs[cur_player] = 1
                    ent_reward = -self.temp * np.log(p) * coefs
            else:
                child_values = np.zeros(self.history.num_players())
                ent_reward = 0
            self.action_values[a] = child_values
            self.history_values += p * (child_values + ent_reward)
