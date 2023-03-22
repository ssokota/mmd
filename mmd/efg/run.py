"""Interface for learning in extensive-form games
"""

import pandas as pd

from .learners import Learner
from ..utils import schedule


def main(
    learner: Learner,
    num_iterations: int,
    fn: str,
) -> None:
    """Run `learner` for `num_iterations`

    Args:
        learner: Learner to run
        num_iterations: Number of iterations to run
        fn: Filename to which to save data
    """
    df = pd.DataFrame({})
    for _, should_save in schedule(num_iterations):
        learner.update()
        if should_save:
            df = pd.concat([df, pd.DataFrame(learner.log_info())], ignore_index=True)
            df.to_csv(fn + ".csv")
