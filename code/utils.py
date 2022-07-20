from typing import NewType
import numpy as np

Opinion = NewType('Opinion', float)
ListOpinion = NewType('ListOpinion', np.ndarray)
Reward = NewType('Reward', float)
ListReward = NewType('ListReward', np.ndarray)
Recommendation = NewType('Recommendation', float)
ListRecommendation = NewType('ListRecommendation', np.ndarray)

KEY_OPINION = 'opinion'
KEY_REWARD = 'reward'
KEY_RECOMMENDATION = 'recommendation'
