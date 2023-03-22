"""Learners for normal-form games.
"""

from typing import Callable, Protocol

import numpy as np

from ..utils import project


class Learner(Protocol):
    """Protocol class for learners"""

    def update(self, payoffs: np.ndarray) -> None:
        """Update internal state of learner using `payoffs`"""

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""

    def test_policy(self) -> np.ndarray:
        """Return policy for testing"""


class MMD(Learner):
    def __init__(
        self,
        n_actions: int,
        initializer: Callable[[int], np.ndarray],
        lr: Callable[[int], float],
        temp: Callable[[int], float],
        magnet_initializer: Callable[[int], np.ndarray],
        magnet_lr: Callable[[int], float],
    ):
        """Implements of magnetic mirror descent

        Args:
            n_actions: Number of actions for the learner
            initializer: Initializer for learner's policy
            lr: Stepsize schedule
            temp: Regularization temperature schedule
            magnet_initializer: Initializer for learner's magnet
            magnet_lr: Stpesize schedule for magnet

        Attributes:
            n_actions
            initializer
            lr
            temp
            magnet_initializer
            magnet_lr
            policy: Current policy
            iteration: Number of completed updates
        """
        self.initializer = initializer
        self.lr = lr
        self.temp = temp
        self.magnet_initializer = magnet_initializer
        self.magnet = project(magnet_initializer(n_actions))
        self.magnet_lr = magnet_lr
        self.policy: np.ndarray = project(initializer(n_actions))
        self.iteration: int = 0

    def update(self, payoffs: np.ndarray) -> None:
        """Update internal state of learner using `payoffs`"""
        self.iteration += 1
        lr = self.lr(self.iteration)
        temp = self.temp(self.iteration)
        mag_lr = self.magnet_lr(self.iteration)
        policy = project(
            np.power(
                self.policy * np.exp(lr * payoffs) * (self.magnet ** (lr * temp)),
                1 / (1 + lr * temp),
            )
        )
        magnet = project(
            np.power(self.policy, mag_lr) * np.power(self.magnet, 1 - mag_lr)
        )
        self.policy = policy
        self.magnet = magnet

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""
        return self.policy.copy()

    def test_policy(self) -> np.ndarray:
        """Return policy for testing"""
        return self.policy.copy()
