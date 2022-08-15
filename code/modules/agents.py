import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import List
from modules.utils import add_hist
from modules.utils import Opinion, ListOpinion, Reward, ListReward, Recommendation, ListRecommendation
from modules.utils import KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD
from modules.utils import KEY_AVERAGE_OPINION, KEY_AVERAGE_RECOMMENDATION, KEY_AVERAGE_REWARD, KEY_STD_OPINION
from modules.parameters import ParametersUser, ParametersPopulation
from modules.trajectory import Trajectory


class OpinionDynamicsEntity(ABC):
    def __init__(self,
                 save_history: bool) -> None:
        self.save_history = save_history

    @abstractmethod
    def initialize(self, initial_state) -> None:
        pass

    @abstractmethod
    def update_state(self, recommendation):
        return

    @abstractmethod
    def plot(self, save: bool = False, name: str = 'sim') -> None:
        return


class User(OpinionDynamicsEntity):
    def __init__(self,
                 initial_state: Opinion,
                 parameters: ParametersUser,
                 save_history: bool) -> None:
        super().__init__(save_history=save_history)
        self._parameters = parameters
        self._x = None
        self.trajectory = Trajectory([KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION])
        self.initialize(initial_state)

    def get_parameters(self) -> ParametersUser:
        return self._parameters

    def initialize(self, initial_state: Opinion) -> None:
        self._x = initial_state
        if self.save_history:
            self.trajectory.append(KEY_OPINION, initial_state)

    def get_opinion(self) -> Opinion:
        return self._x

    def get_reward(self, recommendation: Recommendation) -> Reward:
        opinion_distance = np.abs(self.get_opinion() - recommendation)
        return self.get_parameters().reward(opinion_distance)

    def update_state(self, recommendation: Recommendation) -> Reward:
        reward = self.get_reward(recommendation)
        x_new = self.get_parameters().weight_prejudice*self.get_parameters().prejudice \
                + self.get_parameters().weight_current_opinion*self.get_opinion() \
                + self.get_parameters().weight_recommendation*recommendation
        self._x = Opinion(x_new)
        if self.save_history:
            self.trajectory.append(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION],
                                   items=[x_new, reward, recommendation])
        return reward

    def plot(self, save: bool = False, name: str = 'sim') -> None:
        if not self.save_history:
            return
        self.trajectory.plot(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION],
                             save=save,
                             save_name=name)


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
        self.trajectory = Trajectory([KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD, KEY_AVERAGE_OPINION,
                                      KEY_STD_OPINION, KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION])
        self.initialize(initial_state)

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
            self.trajectory.append(keys=[KEY_OPINION, KEY_AVERAGE_OPINION, KEY_STD_OPINION],
                                   items=[self.get_opinion_vector(), self.average_opinion(), self.std_opinion()])

    def update_state(self, recommendation: ListRecommendation) -> ListReward:
        # TODO: this function is quite inefficient, it could be done with matrices (but it is not so elegant anymore)
        reward = []
        for count, u in enumerate(self.users()):
            reward.append(u.update_state(recommendation=Recommendation(recommendation[count])))
        reward = ListReward(np.asarray(reward))
        self._x = self.get_opinion_vector()
        if self.save_history:
            self.trajectory.append(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION, KEY_AVERAGE_OPINION,
                                         KEY_STD_OPINION, KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION],
                                   items=[self.get_opinion_vector(), reward, recommendation, self.average_opinion(),
                                          self.std_opinion(), np.mean(reward), np.mean(recommendation)])
        return reward

    def get_opinion_vector(self) -> ListOpinion:
        return ListOpinion(np.asarray([u.get_opinion() for u in self.users()]))

    def average_opinion(self) -> Opinion:
        return Opinion(np.average(self.get_opinion_vector()))

    def variance_opinion(self) -> Opinion:
        return Opinion(float(np.var(self.get_opinion_vector())))

    def std_opinion(self) -> Opinion:
        return np.sqrt(self.variance_opinion())

    def __add__(self, other):  # TODO: define sum
        raise ValueError('Not implemented yet.')

    def __eq__(self, other): # TODO: implement
        raise ValueError('Not implemented yet.')

    def plot(self, save: bool = False, name: str = 'sim', intermediate = 100) -> None:
        if not self.save_history:
            return
        """
        if self.n_agents() <= 2:
            for i, u in enumerate(self.users()):
                u.plot(save=save,
                       name='name_' + 'user' + str(i) + '_')
        """
        self.trajectory.plot(keys=[KEY_AVERAGE_OPINION, KEY_STD_OPINION, KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION],
                             save=save,
                             save_name=name)
        # TODO: implement plot Jules
#         plt.scatter(self.trajectory.get_item(key=KEY_OPINION)[0], self.trajectory.get_item(key=KEY_OPINION)[1],
#                     c='blue', s=5)
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.scatter(self.trajectory.get_item(key=KEY_OPINION)[0], self.trajectory.get_item(key=KEY_OPINION)[-1], 1)
        plt.xlabel('Initial Opinion')
        plt.ylabel('Final Opinion')
        
        ax1.plot([0, 1], [0, 1], 'r--', transform=ax1.transAxes, )
        add_hist(ax1, self.trajectory.get_item(key=KEY_OPINION)[0], self.trajectory.get_item(key=KEY_OPINION)[-1])
        plt.show()
        if save:
            plt.savefig(name + '_opinions.png')

        plt.hist(self.trajectory.get_item(key=KEY_OPINION)[0],   density=True, alpha=0.7, label='Initial Opinion')
        plt.hist(self.trajectory.get_item(key=KEY_OPINION)[intermediate], density=True, alpha=0.7, label='Intermediate Opinion')
        plt.hist(self.trajectory.get_item(key=KEY_OPINION)[-1],  density=True, alpha=0.7, label='Final Opinion')
        plt.legend()
        plt.show()
        

class PopulationIdentical(Population):
    def __init__(self,
                 initial_state: ListOpinion,
                 parameters: ParametersPopulation,
                 save_history: bool) -> None:
        super().__init__(initial_state=initial_state,
                         parameters=parameters,
                         save_history=save_history)
        # TODO: implement this
