from abc import ABC
import numpy as np
from typing import NewType


OpinionType = np.ndarray
Opinion = NewType('Opinion', OpinionType)

RewardType = np.ndarray
Reward = NewType('Reward', RewardType)

RecommendationType = np.ndarray
Recommendation = NewType('Recommendation', RecommendationType)
