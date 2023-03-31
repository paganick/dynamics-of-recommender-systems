import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from abc import ABC, abstractmethod
from typing import List, Tuple
import matplotlib.pyplot as plt
from modules.basic import Reward, Recommendation
from modules.samplers import SamplerRecommendation
from modules.trajectory import Trajectory
from modules.utils import KEY_BETTER_REWARD, KEY_EXPLORATION
from modules.saveUtils import save_figure


class Algorithm(ABC):
    def __init__(self,
                 n_agents: int or List[int],
                 save_history: bool = False) -> None:
        if isinstance(n_agents, int):
            self._n_agents = [n_agents]
        elif isinstance(n_agents, list):
            self._n_agents = n_agents
        elif isinstance(n_agents, np.ndarray):
            self._n_agents = n_agents.tolist()
        else:
            raise ValueError('Unknown input type, given ' + type(n_agents) + '.')
        self._n_populations = len(self.n_agents())
        self.save_history = save_history
        self.trajectory = []
        if self.save_history:
            for i in range(self.n_populations()):
                self.trajectory.append(Trajectory([KEY_BETTER_REWARD, KEY_EXPLORATION]))
                self.trajectory[i].reset()

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
                 exploration_probability: float or None = None,
                 save_history: bool = False):
        super().__init__(n_agents, save_history=save_history)
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
        self._best_reward_so_far = [Reward(np.nan * np.ones(n)) for n in self.n_agents()]
        self._best_recommendation_so_far = [Recommendation(np.nan * np.ones(n)) for n in self.n_agents()]
        self._last_recommendation = [Recommendation(np.nan * np.ones(n)) for n in self.n_agents()]

    def get_population_index(self, population: int or None = None) -> int:
        if self.n_populations() == 1:
            if population is None or population == 0:
                return 0
            elif isinstance(population, int) and population != 0:
                raise ValueError('Index not valid, there is only one population.')
            else:
                raise ValueError('Unknown input type, given ' + str(population) + '.')
        else:
            if isinstance(population, int) and 0 <= population <= self.n_populations() - 1:
                return population
            else:
                raise ValueError('Unknown input type,  please input an integer larger than 0.')

    def get_best_recommendation_so_far(self, population: int = None, idx: np.ndarray = None) -> Recommendation:
        population = self.get_population_index(population=population)
        if idx is None:
            return self._best_recommendation_so_far[population]
        else:
            return self._best_recommendation_so_far[population][idx]

    def get_best_reward_so_far(self, population: int = None, idx: np.ndarray = None) -> Reward:
        population = self.get_population_index(population=population)
        if idx is None:
            return self._best_reward_so_far[population]
        else:
            return self._best_reward_so_far[population][idx]

    def get_last_recommendation(self, population: int or None = None, idx: np.ndarray or None = None) -> Recommendation:
        population = self.get_population_index(population=population)
        if idx is None:
            return self._last_recommendation[population]
        else:
            return self._last_recommendation[population][idx]

    def set_best_so_far(self, population: int, idx: np.ndarray, new_recommendation: Recommendation,
                        new_reward: Reward) -> None:
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
        r = []  # recommendation
        explored = [False] * self.n_populations()
        for i in range(self.n_populations()):
            if time == 0:
                # initial time
                r.append(self.recommendation_sampler[i].sample(number=self.n_agents(i)))
            elif self.explore(time=time):  # this now explores for all agents in parallel, to update
                # exploration
                r.append(self.recommendation_sampler[i].sample(number=self.n_agents(i)))
                explored[i] = True
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
        percentage_changed = []
        for i in range(self.n_populations()):
            percentage_changed.append(np.where(self._last_recommendation[i] != r[i])[0].size/self.n_agents(population=i))
        self._last_recommendation = r
        # save
        if self.save_history:
            for i, t in enumerate(self.trajectory):
                t.append(keys=[KEY_BETTER_REWARD, KEY_EXPLORATION],
                         items=[percentage_changed[i], explored[i]])
        # output
        if self.n_populations() == 1:
            return r[0]  # do not return a list if one population only
        else:
            return r

    def plot(self, save: bool = False, show: bool = True, name: str = None, folder: str = None) -> None:
        if not self.save_history:
            return
        _, ax_exploration = plt.subplots(nrows=self.n_populations(), ncols=1)
        if self.n_populations() == 1:
            ax_exploration = [ax_exploration]
        for i in range(self.n_populations()):
            t = self.trajectory[i]
            x = np.arange(0, t.get_number_entries_item(key=KEY_BETTER_REWARD))
            ax_exploration[i].plot(x, t[KEY_BETTER_REWARD], marker='o')
            idx = np.where(t[KEY_EXPLORATION] == 1)[0]
            ax_exploration[i].vlines(idx, 0, 1, colors='k', linestyles='dotted')
            #ax_exploration[i].set_xlim(left=x[0], right=x[-1])
            ax_exploration[i].set_ylim(bottom=0.0, top=1)
            if i <= self.n_populations() - 2:
                ax_exploration[i].xaxis.set_ticklabels([])
        if show:
            plt.show()
        if save:
            save_figure(name=name, folder=folder)


