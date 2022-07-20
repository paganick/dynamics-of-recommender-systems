import numpy as np


def reward_exponential(distance: float, alpha: float) -> float:
    return np.exp(-alpha*(distance**2))
