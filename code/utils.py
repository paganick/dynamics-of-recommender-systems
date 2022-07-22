from typing import NewType
import numpy as np

# types
Opinion = NewType('Opinion', float)
ListOpinion = NewType('ListOpinion', np.ndarray)
Reward = NewType('Reward', float)
ListReward = NewType('ListReward', np.ndarray)
Recommendation = NewType('Recommendation', float)
ListRecommendation = NewType('ListRecommendation', np.ndarray)


def covert_opinion_to_list(x: Opinion) -> ListOpinion:
    return ListOpinion(np.asarray([x]))


def covert_list_to_opinion(x: ListOpinion) -> ListOpinion or Opinion:
    if x.size == 1:
        return Opinion(x[0])
    else:
        return x


def covert_reward_to_list(x: Reward) -> ListReward:
    return ListReward(np.asarray([x]))


def covert_list_to_reward(x: ListReward) -> ListReward or Reward:
    if x.size == 1:
        return Reward(x[0])
    else:
        return x


def covert_recommendation_to_list(x: ListRecommendation) -> ListRecommendation:
    return ListRecommendation(np.asarray([x]))


def covert_list_to_recommendation(x: ListRecommendation) -> ListRecommendation or Recommendation:
    if x.size == 1:
        return Recommendation(x[0])
    else:
        return x


# keys
KEY_OPINION = 'opinion'
KEY_STD_OPINION = 'std_opinion'
KEY_AVERAGE_OPINION = 'average_opinion'
KEY_REWARD = 'reward'
KEY_AVERAGE_REWARD = 'average_reward'
KEY_RECOMMENDATION = 'recommendation'
KEY_AVERAGE_RECOMMENDATION = 'average_recommandation'

# keys reward
KEY_SQUARED_EXPONENTIAL_REWARD = 'squared_exponential'
KEY_EXPONENTIAL_REWARD = 'exponential'
KEY_REWARD_TYPE = 'type'
KEY_REWARD_DECAY_PARAMETER = 'decay_parameter'