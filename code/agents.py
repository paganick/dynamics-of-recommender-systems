import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
from utils import Opinion, ListOpinion, Reward, ListReward, Recommendation, ListRecommendation
from utils import KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD
from utils import KEY_AVERAGE_OPINION, KEY_AVERAGE_RECOMMENDATION, KEY_AVERAGE_REWARD, KEY_STD_OPINION
from parameters import ParametersUser, ParametersPopulation


class OpinionDynamicsEntity(ABC):
    def __init__(self,
                 save_history: bool) -> None:
        self.save_history = save_history
        self._trajectory = {}

    @abstractmethod
    def initialize(self, initial_state) -> None:
        pass

    @abstractmethod
    def update_state(self, recommendation):
        return

    @abstractmethod
    def plot(self, save: bool = False, name: str = 'sim') -> None:
        return

    def get_trajectory(self, key: str) -> ListOpinion or ListReward or ListRecommendation or None:
        if key in self._trajectory:
            if key == KEY_OPINION or key == KEY_AVERAGE_OPINION or key == KEY_STD_OPINION:
                return np.asarray(self._trajectory[key])[:-1]  # cut the last entry since the state has one extra
            else:
                return np.asarray(self._trajectory[key])
        else:
            return None


class User(OpinionDynamicsEntity):
    def __init__(self,
                 initial_state: Opinion,
                 parameters: ParametersUser,
                 save_history: bool) -> None:
        super().__init__(save_history=save_history)
        self._parameters = parameters
        self._x = None
        self._trajectory = {KEY_OPINION: [], KEY_REWARD: [], KEY_RECOMMENDATION: []}
        self.initialize(initial_state)

    def get_parameters(self) -> ParametersUser:
        return self._parameters

    def initialize(self, initial_state: Opinion) -> None:
        self._x = initial_state
        if self.save_history:
            self._trajectory[KEY_OPINION] = [initial_state]
            self._trajectory[KEY_REWARD] = []
            self._trajectory[KEY_RECOMMENDATION] = []

    def get_opinion(self) -> Opinion:
        return self._x

    def get_reward(self, recommendation: Recommendation) -> Reward:
        opinion_distance = np.abs(self.get_opinion() - recommendation)
        return self.get_parameters().reward(opinion_distance)

    def update_state(self, recommendation: Recommendation) -> Reward:
        reward = self.get_reward(recommendation)
        x_new = self.get_parameters().weight_prejudice*self.get_parameters().prejudice + self.get_parameters().weight_current_opinion*self.get_opinion() + self.get_parameters().weight_recommendation*recommendation
        self._x = Opinion(x_new)
        if self.save_history:
            self._trajectory[KEY_OPINION].append(x_new)
            self._trajectory[KEY_REWARD].append(reward)
            self._trajectory[KEY_RECOMMENDATION].append(recommendation)
        return reward

    def plot(self, save: bool = False, name: str = 'sim') -> None:
        if not self.save_history or self.get_trajectory(key=KEY_RECOMMENDATION) is None:
            return
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        horizon = len(self.get_trajectory(key=KEY_RECOMMENDATION))
        ax1.plot(np.arange(horizon), self.get_trajectory(key=KEY_OPINION))
        ax1.set_ylabel('Opinion')
        ax2.plot(np.arange(horizon), self.get_trajectory(key=KEY_RECOMMENDATION))
        ax2.set_ylabel('Recommendation')
        ax3.plot(np.arange(horizon), self.get_trajectory(key=KEY_REWARD))
        ax3.set_ylabel('Reward')
        plt.show()
        if save:
            plt.savefig(name + '.png')


class Population(OpinionDynamicsEntity):
    def __init__(self,
                 initial_state: ListOpinion,
                 parameters: ParametersPopulation,
                 save_history: bool) -> None:
        super().__init__(save_history=save_history)
        self._n_agents = parameters.n_agents
        self._users = []
        self._x = []
        for i in range(self.n_agents()):
            self.add_user(User(initial_state=Opinion(initial_state[i]),
                               parameters=parameters.get_parameters_user(idx_user=i),
                               save_history=False))
        self._trajectory = {KEY_AVERAGE_OPINION: [], KEY_STD_OPINION: [],
                            KEY_AVERAGE_REWARD: [], KEY_AVERAGE_RECOMMENDATION: []}
        self.initialize(initial_state)
        # TODO: build parameter vectors

    def n_agents(self) -> int:
        return self._n_agents

    def users(self) -> List[User]:
        return self._users

    def add_user(self, user: User) -> None:
        self._users.append(user)

    def initialize(self, initial_state: ListOpinion) -> None:
        for count, u in enumerate(self.users()):
            u.initialize(initial_state=Opinion(initial_state[count]))
        self._x = self.get_opinion_vector()
        if self.save_history:
            self._trajectory[KEY_AVERAGE_OPINION] = [self.average_opinion()]
            self._trajectory[KEY_STD_OPINION] = [self.std_opinion()]
            self._trajectory[KEY_AVERAGE_REWARD] = []
            self._trajectory[KEY_AVERAGE_RECOMMENDATION] = []

    def update_state(self, recommendation: ListRecommendation) -> ListReward:
        # TODO: this function is quite inefficient, it could be done with matrices (but it is not so elegant anymore)
        reward = []
        for count, u in enumerate(self.users()):
            reward.append(u.update_state(recommendation=Recommendation(recommendation[count])))
        reward = ListReward(np.asarray(reward))
        self._x = self.get_opinion_vector()
        if self.save_history:
            self._trajectory[KEY_AVERAGE_OPINION].append(self.average_opinion())
            self._trajectory[KEY_STD_OPINION].append(self.std_opinion())
            self._trajectory[KEY_AVERAGE_REWARD].append(np.mean(reward))
            self._trajectory[KEY_AVERAGE_RECOMMENDATION].append(np.mean(recommendation))
        return reward

    def get_opinion_vector(self) -> ListOpinion:
        return ListOpinion(np.asarray([u.get_opinion() for u in self.users()]))

    def average_opinion(self) -> Opinion:
        return Opinion(np.average(self.get_opinion_vector()))

    def variance_opinion(self) -> Opinion:
        return Opinion(float(np.var(self.get_opinion_vector())))

    def std_opinion(self) -> Opinion:
        return np.sqrt(self.variance_opinion())

    def plot(self, save: bool = False, name: str = 'sim') -> None:
        if not self.save_history or self.get_trajectory(key=KEY_AVERAGE_RECOMMENDATION) is None:
            return
        """
        if self.n_agents() <= 2:
            for i, u in enumerate(self.users()):
                u.plot(save=save,
                       name='name_' + 'user' + str(i) + '_')
        """
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        horizon = len(self.get_trajectory(key=KEY_AVERAGE_RECOMMENDATION))
        ax1.plot(np.arange(horizon), self.get_trajectory(key=KEY_AVERAGE_OPINION))
        ax1.set_ylabel('Average Opinion')
        ax2.plot(np.arange(horizon), self.get_trajectory(key=KEY_AVERAGE_RECOMMENDATION))
        ax2.set_ylabel('Average Recommendation')
        ax3.plot(np.arange(horizon), self.get_trajectory(key=KEY_AVERAGE_REWARD))
        ax3.set_ylabel('Average Reward')
        plt.show()
        if save:
            plt.savefig(name + '.png')


class PopulationIdentical(Population):
    def __init__(self,
                 initial_state: ListOpinion,
                 parameters: ParametersPopulation,
                 save_history: bool) -> None:
        super().__init__(initial_state=initial_state,
                         parameters=parameters,
                         save_history=save_history)
        # TODO: implement this
