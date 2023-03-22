"""Learners for extensive-form games using OpenSpiel
"""

from collections import defaultdict
from math import prod
from typing import Callable, Optional, Protocol, Union
from statistics import mean

import numpy as np
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.algorithms.cfr import _CFRSolver
from open_spiel.python.policy import TabularPolicy
from pyspiel import Game

from .tree import Node, Objective
from ..utils import kl, project


class Learner(Protocol):
    def update(self) -> None:
        """Perform update for policies, increment `iteration` by one"""

    def test_policy(self) -> TabularPolicy:
        """Return test policy"""

    def log_info(self) -> dict[str, list[Union[float, str]]]:
        """Return relevant learning information"""

    @property
    def game(self) -> Game:
        """Return the game for learning"""

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""


class CFR(Learner):
    def __init__(self, game: Game, use_plus: bool):
        self._game = game
        self.solver = _CFRSolver(
            game,
            regret_matching_plus=use_plus,
            alternating_updates=use_plus,
            linear_averaging=use_plus,
        )
        self._comparator = None
        self._iteration = 0

    def update(self) -> None:
        """Perform update for policies, increment `iteration` by one"""
        self.solver.evaluate_and_update_policy()
        self._iteration += 1

    def test_policy(self) -> np.ndarray:
        """Return test policy"""
        return self.solver.average_policy()

    def log_info(self) -> dict[str, list[Union[float, str]]]:
        """Return relevant learning information"""
        return {
            "Iteration": [self.iteration],
            "Exploitability": [exploitability(self.game, self.test_policy())],
        }

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""
        return self._comparator

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration


class MMD(Learner):
    def __init__(
        self,
        game: Game,
        temp: Callable[[int], float],
        lr: Callable[[int], float],
        mag_lr: Callable[[int], float],
        objective: Objective,
        comparator: Optional[np.ndarray] = None,
    ):
        """Magnetic mirror descent learner with Q-value feedback

        Args:
            game: OpenSpiel game instance
            temp: Regularization temperature schedule
            lr: Stepsize schedule
            mag_lr: Stepsize schedule for magnet
            objective: Which objective to use
            comparator: Policy from which to compute KL divergence

        Attribues:
            _game
            temp
            lr
            mag_lr
            objective
            _comparator
            _iteration: Number of updates so far
            policy: Current policy
            magnet: Magnet policy
            regret: Mapping from infostates to regrets for last iteration
        """
        self._game = game
        self.temp = temp
        self.lr = lr
        self.mag_lr = mag_lr
        self.objective = objective
        self._comparator = comparator
        self._iteration = 0
        self.policy: TabularPolicy = TabularPolicy(game)
        self.magnet: TabularPolicy = TabularPolicy(game)
        self.regret: dict[str, float] = {}

    def __build_tree(
        self,
    ) -> tuple[
        defaultdict[str, list[dict[int, float]]],
        defaultdict[str, list[float]],
    ]:
        """Build the game tree

        Returns:
            infostate_values: Mapping from infostates to history action values
            reach_probs: Mapping from infostates to history reach probs
        """
        infostate_values = defaultdict(list)
        reach_probs = defaultdict(list)
        queue = [
            Node(
                self.game.new_initial_state(),
                self.policy,
                {i: 1 for i in range(-1, self.game.num_players())},
                self.temp(self.iteration),
                self.objective,
            )
        ]
        while len(queue) > 0:
            node = queue.pop()
            if node.history.is_terminal():
                continue
            if not node.history.is_chance_node():
                cur_player = node.history.current_player()
                infostate = node.history.information_state_string(cur_player)
                acting_qs = {a: q[cur_player] for a, q in node.action_values.items()}
                infostate_values[infostate].append(acting_qs)
                reach_conts = node.reach_contributions.values()
                reach_probs[infostate].append(prod(list(reach_conts)))
            queue += list(node.children.values())
        return infostate_values, reach_probs

    def __get_policy(self, infostate: str, magnet: bool = False) -> np.ndarray:
        """Get the policy at `infostate`

        Args:
            infostate: Infostate for which to get policy
            magnet: Whether to get main policy or magnet policy

        Returns:
            policy at `infostate`
        """
        if magnet:
            pol = self.magnet
        else:
            pol = self.policy
        return pol.action_probability_array[pol.state_lookup[infostate]]

    def __set_policy(
        self, infostate: str, new_policy: Union[dict, np.ndarray], magnet: bool = False
    ) -> None:
        """Set the policy for `infostate` to `new_policy`
        Args:
            infostate: ID for `infostate`
            new_policy: New policy for `infostate`
            magnet: Whether to update magnet policy
        """
        index = self.policy.state_lookup[infostate]
        if isinstance(new_policy, dict):
            pol = np.zeros_like(self.policy.action_probability_array[index])
            for a, p in new_policy.items():
                pol[a] = p
        elif isinstance(new_policy, np.ndarray):
            pol = new_policy
        else:
            raise ValueError
        pol = project(pol)
        if magnet:
            self.magnet.action_probability_array[index] = pol
        else:
            self.policy.action_probability_array[index] = pol

    def __update_infostate(
        self, infostate: str, values: list[dict[int, float]], reach_probs: list[float]
    ) -> None:
        """Update `infostate` given `values` and `reach_probs`
            Also updates regret for `infostate`

        Args:
            infostate: ID for an infostate
            values: History action values for histories in `infostate`'s infoset
            reach_probs: Reach probabilities for histories in `infostate`'s infoset
        """
        expected_values: dict[int, float] = defaultdict(float)
        mu = sum(reach_probs)
        for action_values, p in zip(values, reach_probs):
            for a, q in action_values.items():
                expected_values[a] += q * p / mu
        lr = self.lr(self.iteration)
        temp = self.temp(self.iteration)
        mag_lr = self.mag_lr(self.iteration)
        old_pol = self.__get_policy(infostate)
        mag_pol = self.__get_policy(infostate, magnet=True)
        self.__update_infostate_regret(
            infostate, old_pol, mag_pol, temp, expected_values
        )
        energy = {
            a: (
                np.log(old_pol[a])
                + lr * temp * np.log(mag_pol[a])
                + lr * expected_values[a]
            )
            / (1 + lr * temp)
            for a in expected_values.keys()
        }
        max_energy = max(energy.values())
        prop_policy = {a: np.exp(e - max_energy) for a, e in energy.items()}
        self.__set_policy(infostate, prop_policy)
        prop_mag = np.power(mag_pol, 1 - mag_lr) * np.power(
            self.__get_policy(infostate), mag_lr
        )
        self.__set_policy(infostate, prop_mag, magnet=True)

    def __update_infostate_regret(
        self,
        infostate: str,
        pol: np.ndarray,
        mag_pol: np.ndarray,
        temp: float,
        expected_values: dict[int, float],
    ) -> None:
        """Update the regret for `infostate`
        Args:
            infostate: Infostate for which to update regret
            pol: Policy for `infostate`
            mag_pol: Magnet policy for `infostate`
            temp: Regularization temperature
            expected_values: Action-value feedback for `infostate`
        """
        ev = sum(
            pol[a] * expected_values[a] - temp * pol[a] * (np.log(pol[a] / mag_pol[a]))
            for a in expected_values
        )
        greedy_energy = {
            a: np.log(mag_pol[a]) + expected_values[a] / temp for a in expected_values
        }
        max_greedy_energy = max(greedy_energy.values())
        prop_greedy_policy = {
            a: np.exp(e - max_greedy_energy) for a, e in greedy_energy.items()
        }
        Z = sum(prop_greedy_policy.values())
        greedy_policy = {a: e / Z for a, e in prop_greedy_policy.items()}
        optimal_ev = sum(
            greedy_policy[a] * expected_values[a]
            - temp * greedy_policy[a] * (np.log(greedy_policy[a] / mag_pol[a]))
            for a in expected_values
        )
        self.regret[infostate] = optimal_ev - ev

    def update(self) -> None:
        """Perform update for policies, increment `iteration` by one"""
        self._iteration += 1
        infostate_values, infostate_reach_probs = self.__build_tree()
        for infostate in infostate_values.keys():
            self.__update_infostate(
                infostate, infostate_values[infostate], infostate_reach_probs[infostate]
            )

    def log_info(self) -> dict[str, list[Union[float, str]]]:
        """Return relevant learning information"""
        return {
            "Iteration": [self.iteration],
            "Exploitability": [exploitability(self.game, self.test_policy())],
            "KL Divergence": [
                kl(self.test_policy().action_probability_array, self.comparator)
            ],
            "Regret": [mean(list(self.regret.values()))],
            "Temperature": [self.temp(self.iteration)],
            "Stepsize": [self.lr(self.iteration)],
            "Magnet Stepsize": [self.mag_lr(self.iteration)],
            "Objective": [self.objective.value],
        }

    def test_policy(self) -> TabularPolicy:
        """Return test policy"""
        return self.policy

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""
        return self._comparator

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration
