"""Interface to run experiments for two-player zero-sum normal-form games.
"""

from typing import Union

import pandas as pd

from .game import Game
from .learners import Learner
from ..utils import kl, schedule


def main(
    game: Game,
    maximizer: Learner,
    minimizer: Learner,
    iterations: int,
    fn: str,
    comparators: tuple,
) -> None:
    """Run `maximizer` against `minimizer` in `game`

    Args:
        game: Game in which to train agents
        maximizer: Agent with maximizing objective
        minimizer: Agent with minimizing objective
        iterations: Number of iterations to run
        fn: Filename to which to save data
        comparators: Policies to which to measure KL divergence
    """
    data: dict[str, list[Union[float, str]]] = {
        "Value": [],
        "Metric": [],
        "Iteration": [],
    }
    for i, should_save in schedule(iterations):
        data["Iteration"] += 2 * [i]
        data["Metric"] += ["Exploitability", "KL Divergence"]
        data["Value"] += [
            game.exploitability(maximizer.test_policy(), minimizer.test_policy()),
            (
                kl(maximizer.test_policy(), comparators[0])
                + kl(minimizer.test_policy(), comparators[1])
            )
            / 2,
        ]
        max_payoffs = game.compute_payoffs(None, minimizer.train_policy())
        min_payoffs = game.compute_payoffs(maximizer.train_policy(), None)
        maximizer.update(max_payoffs)
        minimizer.update(min_payoffs)
        if should_save:
            df = pd.DataFrame(data)
            df.to_csv(fn + ".csv")
