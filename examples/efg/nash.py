import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspiel
import seaborn as sns

from mmd.efg.learners import MMD
from mmd.efg.run import main
from mmd.efg.tree import Objective

hyperparameters = {
    "kuhn_poker": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 1 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
    "dark_hex(board_size=2,gameversion=adh)": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 1 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
    "liars_dice(dice_sides=4)": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 1 / np.sqrt(i),
            "lr_schedule": lambda i: 2 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
    "leduc_poker": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 5 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
}

game_choices = [
    "kuhn_poker",
    "dark_hex(board_size=2,gameversion=adh)",
    "liars_dice(dice_sides=4)",
    "leduc_poker",
]

approach_choices = ["annealing_temperature", "moving_magnet"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        choices=game_choices,
        required=True,
    )
    parser.add_argument("--approach", choices=approach_choices, required=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    game = pyspiel.load_game(args.game)
    temp_schedule = hyperparameters[args.game][args.approach]["temp_schedule"]
    lr_schedule = hyperparameters[args.game][args.approach]["lr_schedule"]
    mag_lr_schedule = hyperparameters[args.game][args.approach]["mag_lr_schedule"]
    objective = Objective.standard
    learner = MMD(
        game, temp_schedule, lr_schedule, mag_lr_schedule, objective, comparator=None
    )
    num_iterations = 100_000
    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/efg"
    if args.dry_run:
        num_iterations = 1
        directory += "_dry_run"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"/nash_{args.game}_{args.approach}"
    main(
        learner,
        num_iterations,
        fn,
    )
    df = pd.read_csv(fn + ".csv")
    sns.lineplot(
        data=df,
        x="Iteration",
        y="Exploitability",
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(fn + ".png")
