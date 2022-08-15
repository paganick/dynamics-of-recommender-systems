import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from utils import Reward, ListReward, Recommendation, ListRecommendation
from utils import convert_list_to_recommendation, convert_reward_to_list


class Algorithm(ABC):
    def __init__(self,
                 n_agents: int) -> None:
        self._n_agents = n_agents

    def n_agents(self) -> int:
        return self._n_agents

    @abstractmethod
    def compute_recommendation(self,
                               reward: None or Reward or List[Reward],
                               time: int) -> Recommendation or ListRecommendation:
        return None


class UtilityMatrix(Algorithm):
    def __init__(self,
                 n_agents: int,
                 exploration_frequency: int or None = None,
                 exploration_probability: float or None = None):
        super().__init__(n_agents)
        self._best_reward_so_far = ListReward(np.zeros(self._n_agents))
        self._best_recommendation_so_far = ListRecommendation(np.empty(self._n_agents))
        self._last_recommendation = ListRecommendation(np.empty(self._n_agents))
        self.exploration_frequency = exploration_frequency
        self.exploration_probability = exploration_probability

    def get_best_recommendation_so_far(self, idx: np.ndarray = None) -> ListRecommendation:
        if idx is None:
            return ListRecommendation(self._best_recommendation_so_far)
        else:
            return ListRecommendation(self._best_recommendation_so_far[idx])

    def get_best_reward_so_far(self, idx: np.ndarray = None) -> ListReward:
        if idx is None:
            return ListReward(self._best_reward_so_far)
        else:
            return ListReward(self._best_reward_so_far[idx])

    def get_last_recommendation(self, idx: np.ndarray = None) -> ListRecommendation:
        if idx is None:
            return ListRecommendation(self._last_recommendation)
        else:
            return ListRecommendation(self._last_recommendation[idx])

    def set_best_so_far(self, idx: np.ndarray, new_recommendation: ListRecommendation, new_reward: ListReward) -> None:
        assert idx.size == new_recommendation.size, 'The size must coincide.'
        assert idx.size == new_reward.size, 'The size must coincide.'
        self._best_recommendation_so_far[idx] = new_recommendation
        self._best_reward_so_far[idx] = new_reward

    def explore(self, time: int) -> bool:
        explore = False
        if self.exploration_frequency is not None:
            explore = explore or (time % self.exploration_frequency == 0)
        if self.exploration_probability is not None:
            explore = explore or float(np.random.rand(1)) <= self.exploration_probability #TO BE CHECKED
        return bool(explore)

    def compute_recommendation(self,
                               reward: None or Reward or ListReward,
                               time: int) -> Recommendation or ListRecommendation:
        if time == 0 or (reward is None and self.get_best_reward_so_far() is None):
            # initial time
            #r = np.random.uniform(low=-1.0, high=1.0, size=self.n_agents())
            r = np.random.normal(loc=0.0, scale=1.0, size=self.n_agents())
        elif self.get_best_reward_so_far().size == 0:
            # if the best is empty, initialize it
            self.set_best_so_far(idx=np.arange(0, self.n_agents()),
                                 new_reward=ListReward(np.asarray([reward])),
                                 new_recommendation=self.get_last_recommendation())
            r = self.get_best_recommendation_so_far()
        elif self.explore(time=time):
            #r = np.random.uniform(low=-1.0, high=1.0, size=self.n_agents())
            r = np.random.normal(loc=0.0, scale=1.0, size=self.n_agents())
        else:
            # no exploration
            if self.n_agents() == 1:
                reward = convert_reward_to_list(reward)
            # find agents for which things improved
            if np.all(self.get_best_reward_so_far() >= reward):
                r = self.get_best_recommendation_so_far()
            else:
                idx_better = np.where(reward > self.get_best_reward_so_far())[0]
                # update best
                if idx_better.size >= 1:
                    self.set_best_so_far(idx=idx_better,
                                         new_recommendation=self.get_last_recommendation(idx=idx_better),
                                         new_reward=reward[idx_better])
                # recommend the best
                r = self.get_best_recommendation_so_far()
        self._last_recommendation = r
        return convert_list_to_recommendation(r)
