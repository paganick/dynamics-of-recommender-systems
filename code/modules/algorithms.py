import numpy as np
from abc import ABC, abstractmethod
from typing import List
from modules.basic import Reward, Recommendation
from modules.samplers import SamplerRecommendation


class Algorithm(ABC):
    def __init__(self,
                 n_agents: int or List[int]) -> None:
        if isinstance(n_agents, int):
            self._n_agents = [n_agents]
        elif isinstance(n_agents, list):
            self._n_agents = n_agents
        elif isinstance(n_agents, np.ndarray):
            self._n_agents = n_agents.tolist()
        else:
            raise ValueError('Unknown input type, given ' + type(n_agents) + '.')
        self._n_populations = len(self.n_agents())

    def n_agents(self, population: int or None = None) -> int or List[int]:
        if population is None:
            return self._n_agents
        else:
            return self._n_agents[population]

    def n_populations(self) -> int:
        return self._n_populations

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def compute_recommendation(self,
                               reward: None or Reward or List[Reward],
                               time: int) -> Recommendation or List[Recommendation]:
        return None


class UtilityMatrix(Algorithm):
    def __init__(self,
                 n_agents: int or List[int],
                 recommendation_sampler: SamplerRecommendation or List[SamplerRecommendation],
                 exploration_frequency: int or None = None,  # TODO: extend to list
                 exploration_probability: float or None = None):
        super().__init__(n_agents)
        self._best_reward_so_far, self._best_recommendation_so_far, self._last_recommendation = None, None, None
        self.reset()
        self.exploration_frequency = exploration_frequency
        self.exploration_probability = exploration_probability
        if isinstance(recommendation_sampler, list) and len(recommendation_sampler) != self.n_populations:
            raise ValueError('The length of recommendation_sampler should coincide with n_populations.')
        elif isinstance(recommendation_sampler, SamplerRecommendation):
            self.recommendation_sampler = [recommendation_sampler for _ in range(self.n_populations())]
        else:
            raise ValueError('Unknown input type.')

    def reset(self) -> None:
        self._best_reward_so_far = [Reward(np.nan*np.ones(n)) for n in self.n_agents()]
        self._best_recommendation_so_far = [Recommendation(np.nan*np.ones(n)) for n in self.n_agents()]
        self._last_recommendation = [Recommendation(np.nan*np.ones(n)) for n in self.n_agents()]

    def get_best_recommendation_so_far(self, population: int = None, idx: np.ndarray = None) -> Recommendation:
        if self.n_populations() != 1 and population is None:
            raise ValueError('Please input population index.')
        if self.n_populations() == 1 and population is None:
            population = 0
        if idx is None:
            return self._best_recommendation_so_far[population]
        else:
            return self._best_recommendation_so_far[population][idx]

    def get_best_reward_so_far(self, population: int = None, idx: np.ndarray = None) -> Reward:
        if self.n_populations() != 1 and population is None:
            raise ValueError('Please input population index.')
        if self.n_populations() == 1 and population is None:
            population = 0
        if idx is None:
            return self._best_reward_so_far[population]
        else:
            return self._best_reward_so_far[population][idx]

    def get_last_recommendation(self, population: int or None = None, idx: np.ndarray or None = None) -> Recommendation:
        if self.n_populations() != 1 and population is None:
            raise ValueError('Please input population index.')
        if self.n_populations() == 1 and population is None:
            population = 0
        if idx is None:
            return self._last_recommendation[population]
        else:
            return self._last_recommendation[population][idx]

    def set_best_so_far(self, population: int, idx: np.ndarray, new_recommendation: Recommendation, new_reward: Reward) -> None:
        assert idx.size == new_recommendation.size, 'The size must coincide.'
        assert idx.size == new_reward.size, 'The size must coincide.'
        self._best_recommendation_so_far[population][idx] = new_recommendation
        self._best_reward_so_far[population][idx] = new_reward

    def explore(self, time: int) -> bool:
        explore = False
        if self.exploration_frequency is not None:
            explore = explore or (time % self.exploration_frequency == 0)
        if self.exploration_probability is not None:
            explore = explore or float(np.random.rand(1)) <= self.exploration_probability  # TO BE CHECKED
        return bool(explore)

    def compute_recommendation(self,
                               reward: None or Reward or List[Reward],
                               time: int) -> Recommendation or List[Recommendation]:
        if not isinstance(reward, list):
            reward = [reward]
        r = []
        for i in range(self.n_populations()):
            if time == 0:
                # initial time
                r.append(self.recommendation_sampler[i].sample(number=self.n_agents(i)))
            elif self.explore(time=time):  # this now explores for all agents in parallel, to update
                # exploration
                r.append(self.recommendation_sampler[i].sample(number=self.n_agents(i)))
            elif np.any(np.isnan(self.get_best_reward_so_far(population=i))):
                # if the best is empty (i.e., nan for at least one user), initialize it
                self.set_best_so_far(idx=np.arange(0, self.n_agents(i)),
                                     population=i,
                                     new_reward=reward[i],
                                     new_recommendation=self.get_last_recommendation(population=i))
                r.append(self.get_best_recommendation_so_far(population=i))
            else:
                # no exploration
                # find agents for which things improved
                if np.all(self.get_best_reward_so_far(population=i) >= reward[i]):  # no agents have improved
                    r.append(self.get_best_recommendation_so_far(population=i))
                else:
                    idx_better = np.where(reward[i] > self.get_best_reward_so_far(population=i))[0]
                    # update best
                    if idx_better.size >= 1:
                        self.set_best_so_far(idx=idx_better,
                                             population=i,
                                             new_recommendation=self.get_last_recommendation(population=i,
                                                                                             idx=idx_better),
                                             new_reward=reward[i][idx_better])
                    # recommend the best
                    r.append(self.get_best_recommendation_so_far(population=i))
        # update last recommendation
        self._last_recommendation = r
        # output
        if self.n_populations() == 1:
            return r[0]  # do not return a list if one population only
        else:
            return r
