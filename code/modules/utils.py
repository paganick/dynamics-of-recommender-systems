from typing import NewType
import numpy as np

""""
Opinion = NewType('Opinion', float)
ListOpinion = NewType('ListOpinion', np.ndarray)
Reward = NewType('Reward', float)
ListReward = NewType('ListReward', np.ndarray)
Recommendation = NewType('Recommendation', float)
ListRecommendation = NewType('ListRecommendation', np.ndarray)


def convert_opinion_to_list(x: Opinion) -> ListOpinion:
    return ListOpinion(np.asarray([x]))


def convert_list_to_opinion(x: ListOpinion) -> ListOpinion or Opinion:
    if x.size == 1:
        return Opinion(x[0])
    else:
        return x


def convert_reward_to_list(x: Reward) -> ListReward:
    return ListReward(np.asarray([x]))


def convert_list_to_reward(x: ListReward) -> ListReward or Reward:
    if x.size == 1:
        return Reward(x[0])
    else:
        return x


def convert_recommendation_to_list(x: ListRecommendation) -> ListRecommendation:
    return ListRecommendation(np.asarray([x]))


def convert_list_to_recommendation(x: ListRecommendation) -> ListRecommendation or Recommendation:
    if x.size == 1:
        return Recommendation(x[0])
    else:
        return x
"""

# keys
KEY_ITERATION = 'iteration'
KEY_OPINION = 'opinion'
KEY_STD_OPINION = 'std_opinion'
KEY_AVERAGE_OPINION = 'average_opinion'
KEY_REWARD = 'reward'
KEY_AVERAGE_REWARD = 'average_reward'
KEY_RECOMMENDATION = 'recommendation'
KEY_AVERAGE_RECOMMENDATION = 'average_recommendation'

# key input output
KEY_N_AGENTS = 'n_agents'
KEY_USER = 'user'
KEY_PARAMETERS_USERS = 'parameter_users'

# keys reward
KEY_SQUARED_EXPONENTIAL_REWARD = 'squared_exponential'
KEY_EXPONENTIAL_REWARD = 'exponential'
KEY_REWARD_TYPE = 'type'
KEY_REWARD_DECAY_PARAMETER = 'decay_parameter'
KEY_BETTER_REWARD = 'got_better_reward'
KEY_EXPLORATION = 'explored'

# keys recommendation sampler
KEY_SAMPLER_TYPE = 'type'
KEY_SAMPLER_OBJECT = 'object'
KEY_SAMPLER_OBJECT_RECOMMENDATION = 'recommendation'
KEY_SAMPLER_OBJECT_OPINION = 'opinion'
KEY_SAMPLER_TYPE_UNIFORM = 'uniform'
KEY_SAMPLER_TYPE_GAUSSIAN = 'gaussian'
KEY_SAMPLER_TYPE_MIXTURE_GAUSSIAN = 'mixture of gaussian'
KEY_SAMPLER_UNIFORM_LOW = 'low'
KEY_SAMPLER_UNIFORM_HIGH = 'high'
KEY_SAMPLER_GAUSSIAN_MEAN = 'mean'
KEY_SAMPLER_GAUSSIAN_STD = 'std'
KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN = 'mean'
KEY_SAMPLER_MIXTURE_GAUSSIAN_STD = 'std'
