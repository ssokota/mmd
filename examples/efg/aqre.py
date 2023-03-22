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
from mmd.utils import project

game_choices = [
    "kuhn_poker",
    "dark_hex(board_size=2,gameversion=adh)",
    "liars_dice(dice_sides=4)",
    "leduc_poker",
]
temperature_choices = [0.5, 0.2, 0.1, 0.05]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        choices=game_choices,
        required=True,
    )
    parser.add_argument(
        "--temperature", choices=temperature_choices, type=float, required=True
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    learner = MMD(
        game=pyspiel.load_game(args.game),
        temp=lambda i: args.temperature,
        lr=lambda i: args.temperature / 10,
        mag_lr=lambda i: 0,
        objective=Objective.standard,
        comparator=np.apply_along_axis(
            func1d=project,
            axis=-1,
            arr=np.load(f"{here}/aqres/{args.game}_{args.temperature}.npy"),
        ),
    )
    num_iterations = 50_000
    directory = f"{here}/../../results/efg"
    if args.dry_run:
        num_iterations = 1
        directory += "_dry_run"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"/aqre_{args.game}_{args.temperature}"
    main(learner, num_iterations, fn)
    df = pd.read_csv(fn + ".csv")
    sns.lineplot(
        data=df,
        x="Iteration",
        y="KL Divergence",
    )
    plt.yscale("log")
    plt.ylim(bottom=1e-6)  # pre-computed solution precision
    plt.savefig(fn + ".png")
