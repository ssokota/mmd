"""Utility functions and classes
"""

from typing import Optional

import numpy as np

SMALL_POSITIVE = 1e-10
DEFAULT_VALUE = 1


def project(x: np.ndarray) -> np.ndarray:
    """Project `x` to simplex, enforces minimum value for numerical stability"""
    assert np.all(np.logical_or(x > 0, np.isclose(x, 0)))
    x = np.maximum(x, SMALL_POSITIVE)
    return x / x.sum()


def kl(
    x: np.ndarray,
    y: Optional[np.ndarray],
) -> float:
    """Return KL divergence between `x` and `y`
        If `y` is None returns `DEFAULT_VALUE`

    Args:
        x: First argument for KL
        y: Second argument for KL
    """
    if y is None:
        return DEFAULT_VALUE
    assert x.shape == y.shape
    cum = 0
    for x_, y_ in zip(x.flatten(), y.flatten()):
        if np.isclose(x_, 0):
            continue
        if np.isclose(y_, 0):
            cum += x_ * np.log(x_ / SMALL_POSITIVE)
        else:
            cum += x_ * np.log(x_ / y_)
    return float(cum / np.prod(x.shape[:-1]))


def is_power_of_2(n: int) -> bool:
    """Return whether `n` is power of 2"""
    return (n & (n - 1) == 0) and n != 0


def schedule(upper_lim: int) -> list[tuple[int, bool]]:
    """Return schedule of iterations and whether to save data"""
    ls: list[tuple[int, bool]] = []
    for i in range(upper_lim):
        ls.append((i, is_power_of_2(i) or i == upper_lim - 1))
    return ls
