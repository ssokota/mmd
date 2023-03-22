import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mmd.nfg.game import Game
from mmd.nfg.learners import MMD
from mmd.nfg.run import main
from mmd.utils import project


normalize = lambda x: (x - x.min()) / (x.max() - x.min())
game = Game(normalize(np.array([[0, -3, 3], [3, 0, -1], [-3, 1, 0]])))
n_actions = game.payoff_table.shape

# Pre-computed solutions
qres = {
    0.05: (
        project(np.array([0.1517447, 0.46794775, 0.38030755])),
        project(np.array([0.1517447, 0.46794775, 0.38030755])),
    ),
    0.1: (
        project(np.array([0.17347803, 0.48630324, 0.34021873])),
        project(np.array([0.17347803, 0.48630324, 0.34021873])),
    ),
    0.2: (
        project(np.array([0.22364492, 0.47798817, 0.29836691])),
        project(np.array([0.22364492, 0.47798817, 0.29836691])),
    ),
    0.5: (
        project(np.array([0.29813319, 0.41382045, 0.28804635])),
        project(np.array([0.29813319, 0.41382045, 0.28804635])),
    ),
}

temperature_choices = [0.5, 0.2, 0.1, 0.05]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature", choices=temperature_choices, type=float, required=True
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    args = parser.parse_args()
    comparators = qres[args.temperature]
    players = [
        MMD(
            n_actions=n_action,
            initializer=lambda t: np.ones(n_action) / n_action,
            lr=lambda t: args.temperature,
            temp=lambda t: args.temperature,
            magnet_initializer=lambda t: np.ones(n_action) / n_action,
            magnet_lr=lambda t: 0,
        )
        for n_action in n_actions
    ]
    iterations = 10_000
    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/nfg"
    if args.dry_run:
        iterations = 1
        directory += "_dry_run"
    Path(directory).mkdir(exist_ok=True, parents=True)
    fn = directory + f"/perturbed_rps_{args.temperature}"
    main(
        game=game,
        maximizer=players[0],
        minimizer=players[1],
        iterations=10_000,
        fn=fn,
        comparators=comparators,
    )
    df = pd.read_csv(fn + ".csv")
    g = sns.relplot(
        data=df,
        x="Iteration",
        y="Value",
        col="Metric",
        kind="line",
        facet_kws=dict(sharey=False, sharex=False),
    )
    g.set(yscale="log")
    g.axes[0, 0].set_xscale("log")
    g.axes[0, 1].set_ylim(bottom=1e-6)
    plt.tight_layout()
    plt.savefig(fn + ".png")
