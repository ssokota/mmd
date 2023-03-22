import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspiel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import seaborn as sns
import torch

from mmd.h2h.run import main


def make_player(here: str, game: str, agent: str, seed: int):
    if "mmd" in agent or "ppo" in agent:
        return player_factory(
            torch.load(f"{here}/models/{agent[:3]}_{game}_{agent[4:]}_{seed}.pt")
        )
    elif agent == "random":
        return random
    elif agent == "arbitrary":
        return arbitrary
    raise ValueError


def random(infostate: list[int], legal_mask: list[int]):
    probs = np.array(legal_mask) / sum(legal_mask)
    return np.random.choice(range(len(legal_mask)), p=probs)


def arbitrary(infostate: list[int], legal_mask: list[int]):
    return np.argmax(legal_mask)


def player_factory(policy: FullyConnectedNetwork):
    def player(infostate: list[int], legal_mask: list[int]):
        x = torch.Tensor(infostate).view(1, -1)
        with torch.no_grad():
            assert policy._logits is not None
            logits = policy._logits(policy._hidden_layers(x))
        unnorm_decision_rule = (
            torch.nn.functional.softmax(logits, dim=-1).view(-1)
            * torch.Tensor(legal_mask).int()
        )
        decision_rule = unnorm_decision_rule / unnorm_decision_rule.sum()
        action = np.random.choice(
            range(decision_rule.shape[-1]), p=decision_rule.numpy()
        )
        return action

    return player


game_choices = ["dark_hex(board_size=3,gameversion=adh)", "phantom_ttt"]
agent_choices = ["mmd_1m", "mmd_10m", "ppo_1m", "ppo_10m", "random", "arbitrary"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        choices=game_choices,
        required=True,
    )
    parser.add_argument(
        "--agent1",
        choices=agent_choices,
        required=True,
    )
    parser.add_argument(
        "--agent2",
        choices=agent_choices,
        required=True,
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    game = pyspiel.load_game(args.game)
    seeds = [0, 1, 2]
    here = os.path.dirname(os.path.abspath(__file__))
    agents = [
        [make_player(here, args.game, agent, s) for s in seeds]
        for agent in [args.agent1, args.agent2]
    ]
    num_episodes = 1_000
    directory = f"{here}/../../results/h2h"
    if args.dry_run:
        num_episodes = 2
        directory += "_dry_run"
    Path(directory).mkdir(parents=True, exist_ok=True)
    p1_outcomes = []
    moving = []
    for s1, a1 in enumerate(agents[0]):
        for s2, a2 in enumerate(agents[1]):
            # agent 1 moves first
            fn1 = directory + f"/{args.game}_{args.agent1}_{s1}_{args.agent2}_{s2}"
            main(game, [a1, a2], num_episodes // 2, fn1)
            p1_outcomes += np.load(fn1 + ".npy")[:, 0].tolist()
            moving += (num_episodes // 2) * ["First Moving"]
            # agent 1 moves second
            fn2 = directory + f"/{args.game}_{args.agent2}_{s2}_{args.agent1}_{s1}"
            main(game, [a2, a1], num_episodes // 2, fn2)
            p1_outcomes += np.load(fn2 + ".npy")[:, 1].tolist()
            moving += (num_episodes // 2) * ["Second Moving"]
    colname = f"{args.agent1} Return Against {args.agent2}"
    df = pd.DataFrame({colname: p1_outcomes, "Order": moving})
    sns.countplot(data=df, x=colname, hue="Order")
    plt.savefig(directory + f"/{args.game}_{args.agent1}_{args.agent2}.png")
    expected_return = round(df[colname].mean(), 2)
    std_err = round(df[colname].std() / np.sqrt(len(df[colname])), 2)
    print(f"Expected {colname}: {expected_return} +/- {std_err}")
