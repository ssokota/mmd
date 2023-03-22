"""Interface for running head-to-head evaluations.
"""

from typing import Callable

import numpy as np
import pyspiel


def main(
    game: pyspiel.Game,
    players: list[Callable[[list[float], list[int]], int]],
    num_episodes: int,
    fn: str,
) -> None:
    """Evaluate `players` against one another in `game`

    Args:
        game: Game in which to evaluate `players`
        players: Players to evaluate
        num_episodes: Number of episodes for which to evalute `players`
        fn: Filename to which to save data
    """
    stats = []
    for _ in range(num_episodes):
        history = game.new_initial_state()
        while not history.is_terminal():
            if history.is_chance_node():
                action_list, prob_list = zip(*history.chance_outcomes())
                action = np.random.choice(action_list, p=prob_list)
            else:
                player = players[history.current_player()]
                action = player(
                    history.information_state_tensor(), history.legal_actions_mask()
                )
            history.apply_action(action)
        stats.append(history.returns())
    np.save(fn + ".npy", np.array(stats))
