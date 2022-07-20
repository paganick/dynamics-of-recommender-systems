import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
from utils import Opinion, ListOpinion, Reward, ListReward, Recommendation, ListRecommendation
from utils import KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD


class User(ABC):
    def __init__(self,
                 initial_state: Opinion,
                 prejudice: Opinion,
                 weight_prejudice: float,
                 weight_current_opinion: float,
                 weight_recommendation: float,
                 reward: Callable,
                 save_history: bool) -> None:
        # state of the user
        self._x = None
        self._trajectory = {'opinion': None, 'reward': None, 'recommendation': None}
        # parameters
        self.alpha = weight_prejudice
        self.beta = weight_current_opinion
        self.gamma = weight_recommendation
        assert self.alpha + self.beta + self.gamma == 1, 'Invalid input parameters, they should sum to 1.'
        self.prejudice = prejudice
        self.save_history = save_history
        self.reward_function = reward
        # initialize
        self.initialize(initial_state)

    def initialize(self, initial_state: Opinion) -> None:
        self._x = initial_state
        if self.save_history:
            self._trajectory[KEY_OPINION] = [initial_state]
            self._trajectory[KEY_REWARD] = []
            self._trajectory[KEY_RECOMMENDATION] = []

    def get_opinion(self) -> Opinion:
        return self._x

    def get_trajectory(self, key: str) -> ListOpinion or ListReward or ListRecommendation or None:
        if key.casefold() in self._trajectory:
            if key.casefold() == KEY_OPINION.casefold():
                return np.asarray(self._trajectory[key])[:-1]  # cut the last entry since the state has one extra
            else:
                return np.asarray(self._trajectory[key])
        else:
            return None

    def update_state(self, recommendation: Recommendation) -> Reward:
        reward = self.reward_function(np.abs(self.get_opinion() - recommendation)) #TODO: reward here ok?
        reward = Reward(reward)
        x_new = self.alpha*self.prejudice + self.beta*self.get_opinion() + self.gamma*recommendation
        self._x = Opinion(x_new)
        if self.save_history:
            self._trajectory[KEY_OPINION].append(x_new)
            self._trajectory[KEY_REWARD].append(reward)
            self._trajectory[KEY_RECOMMENDATION].append(recommendation)
        return reward

    def plot(self) -> None:
        if not self.save_history:
            return
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        horizon = len(self.get_trajectory(key=KEY_RECOMMENDATION))
        ax1.plot(np.arange(horizon), np.asarray(self.get_trajectory(key=KEY_OPINION)))
        ax1.set_ylabel('Opinion')
        ax2.plot(np.arange(horizon), np.asarray(self.get_trajectory(key=KEY_RECOMMENDATION)))
        ax2.set_ylabel('Recommendation')
        ax3.plot(np.arange(horizon), np.asarray(self.get_trajectory(key=KEY_REWARD)))
        ax3.set_ylabel('Reward')


class Population(ABC): #TODO: implement this
    def __init__(self) -> None:
        return None

    def average_opinion(self) -> None:
        return None

    def variance_opinion(self) -> None:
        return None

    def std_opinion(self) -> None:
        return np.sqrt(self.variance_opinion())