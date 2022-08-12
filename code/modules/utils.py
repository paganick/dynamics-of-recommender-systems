from typing import NewType
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# types
Opinion = NewType('Opinion', float)
ListOpinion = NewType('ListOpinion', np.ndarray)
Reward = NewType('Reward', float)
ListReward = NewType('ListReward', np.ndarray)
Recommendation = NewType('Recommendation', float)
ListRecommendation = NewType('ListRecommendation', np.ndarray)


def add_hist(ax, x, y, c='blue', legend=''):
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1, pad=0.15, sharex=ax)
        axHisty = divider.append_axes("right", 1, pad=0.2, sharey=ax)
        # make some labels invisible
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHisty.yaxis.set_tick_params(labelleft=False)
        # now determine nice limits by hand:
        binwidth = 0.1
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1)*binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        axHistx.hist(x, density=True, bins=bins, color=c)
        # axHistx.set_title(legend)
        axHisty.hist(y, density=True, bins=bins, orientation='horizontal', color=c)


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

# keys recommendation sampler
KEY_SAMPLER_RECOMMENDATION_TYPE = 'type'
KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW = 'low'
KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH = 'high'
