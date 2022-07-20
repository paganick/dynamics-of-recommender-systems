import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from utils import Reward, ListReward, Recommendation, ListRecommendation


class Algorithm(ABC):
    def __init__(self,
                 n_agents: int) -> None:
        self._n_agents = n_agents

    def n_agents(self) -> int:
        return self._n_agents

    @abstractmethod
    def compute_recommendation(self,
                               rewards: Reward or List[Reward],
                               time: int) -> Recommendation or ListRecommendation:
        return None


class UtilityMatrix(Algorithm):
    def __init__(self,
                 n_agents: int):
        super().__init__(n_agents)
        self._best_reward_so_far = ListReward(np.empty(self._n_agents))
        self._best_recommendation_so_far = ListRecommendation(np.empty(self._n_agents))
        self._last_recommendation = ListRecommendation(np.empty(self._n_agents))

    def get_best_recommendation_so_far(self, idx: np.ndarray) -> ListRecommendation:
        return ListRecommendation(self._best_recommendation_so_far[idx].copy())

    def get_best_reward_so_far(self, idx: np.ndarray) -> ListReward:
        return ListReward(self._best_reward_so_far[idx].copy())

    def get_last_recommendation(self, idx: np.ndarray) -> ListRecommendation:
        return ListRecommendation(self._last_recommendation[idx].copy())

    def set_best_so_far(self, idx: np.ndarray, new_recommendation: ListRecommendation, new_reward: ListReward) -> None:
        assert idx.size == new_recommendation.size, 'The size must coincide.'
        assert idx.size == new_reward.size, 'The size must coincide.'
        self._best_recommendation_so_far[idx] = new_recommendation
        self._best_reward_so_far[idx] = new_reward

    def compute_recommendation(self,
                               reward: Reward or ListReward,
                               time: int) -> Recommendation or List[Recommendation]:
        if time == 0:
            # initial time
            r = np.random.uniform(low=-1.0, high=1.0, size=self.n_agents())
        elif time % 100 == 0:
            # every 10 explore
            r = np.random.uniform(low=-1.0, high=1.0, size=self.n_agents())
        else:
            # no exploration
            # check if reward are there
            # find agents for which things improved
            if np.all(self.get_best_reward_so_far(idx=np.arange(self.n_agents())) >= reward):
                r = self.get_best_recommendation_so_far(idx=np.arange(self.n_agents()))
            else:
                idx_better = np.where(reward > self.get_best_reward_so_far(idx=np.arange(self.n_agents())))[0]
                # update best
                if idx_better.size >= 1:
                    self.set_best_so_far(idx=idx_better,
                                         new_recommendation=self.get_last_recommendation(idx=idx_better),
                                         new_reward=reward[idx_better])
                # recommend the best
                r = self.get_best_recommendation_so_far(idx=np.arange(self.n_agents()))
        self._last_recommendation = r
        if self.n_agents() == 1:
            return Recommendation(r[0])
        else:
            return r
