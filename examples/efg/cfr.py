import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspiel
import seaborn as sns

from mmd.efg.learners import CFR
from mmd.efg.run import main

game_choices = [
    "kuhn_poker",
    "dark_hex(board_size=2,gameversion=adh)",
    "liars_dice(dice_sides=4)",
    "leduc_poker",
]

variant_choices = ["standard", "plus"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        choices=game_choices,
        required=True,
    )
    parser.add_argument("--variant", choices=variant_choices)
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    game = pyspiel.load_game(args.game)
    learner = CFR(game, args.variant == "plus")
    alg = f"cfr_{args.variant}"
    num_iterations = 100_000
    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/efg"
    if args.dry_run:
        num_iterations = 1
        directory += "_dry_run"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"/{alg}_{args.game}"
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
