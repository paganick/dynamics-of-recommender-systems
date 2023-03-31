import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List
from modules.plotUtils import plot_opinion_shift, plot_opinions_time, plot_sankey_single_population, \
    plot_sankey_multiple_populations
from modules.basic import Opinion, OpinionType, Recommendation, Reward
from modules.utils import KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD
from modules.utils import KEY_AVERAGE_OPINION, KEY_AVERAGE_RECOMMENDATION, KEY_AVERAGE_REWARD, KEY_STD_OPINION
from modules.parameters import ParametersUser, ParametersPopulation
from modules.trajectory import Trajectory
from modules.samplers import SamplerOpinion


class OpinionDynamicsEntity(ABC):
    def __init__(self,
                 initial_state: Opinion or SamplerOpinion or None = None,
                 save_history: bool = False) -> None:
        self.save_history = save_history
        self.trajectory = None
        self._x = None
        self._initial_state = initial_state
        self._last_initial_state = None

    def initial_state(self) -> SamplerOpinion or Opinion:
        return self._initial_state

    def last_initial_state(self) -> SamplerOpinion or Opinion:
        return self._last_initial_state

    @abstractmethod
    def initialize(self, initial_state, initialize_with_prejudice: bool = False) -> None:
        pass

    @abstractmethod
    def update_state(self, recommendation):
        return

    @abstractmethod
    def plot(self, show: bool = True, save: bool = False, name: str = None, folder: str = None) -> tuple:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def save_trajectory_to_file(self, name: str = None, folder: str = None):
        pass


class User(OpinionDynamicsEntity):
    def __init__(self,
                 parameters: ParametersUser,
                 initial_state: Opinion or SamplerOpinion or None = None,
                 initialize_with_prejudice: bool = False,
                 save_history: bool = False) -> None:
        super().__init__(initial_state=initial_state,
                         save_history=save_history)
        self._parameters = parameters
        self.trajectory = Trajectory([KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION])
        self.initialize(initial_state=self.initial_state(),
                        initialize_with_prejudice=initialize_with_prejudice)

    def get_parameters(self) -> ParametersUser:
        return self._parameters

    def initialize(self,
                   initial_state: Opinion or SamplerOpinion or None = None,
                   initialize_with_prejudice: bool = False) -> None:
        if initialize_with_prejudice and initial_state is not None:
            raise ValueError('Either initialize with prejudice or input a given state.')
        elif initialize_with_prejudice:
            self._x = self.get_parameters().prejudice
        elif isinstance(initial_state, SamplerOpinion):
            self._x = initial_state.sample(number=1)
        elif isinstance(initial_state, OpinionType):
            self._x = initial_state
        elif initial_state is None:
            if isinstance(self.initial_state(), SamplerOpinion):
                self._x = self.initial_state().sample(number=1)
            else:
                self._x = self.initial_state()
        else:
            raise ValueError('Unknown input, received ' + type(initial_state).__name__ + '.')
        self._last_initial_state = self.opinion()  # store initial state
        if self.save_history:
            self.trajectory.reset()
            self.trajectory.append(KEY_OPINION, self.opinion())

    def opinion(self) -> Opinion or None:
        return self._x

    def reward(self, recommendation: Recommendation) -> Reward:
        if self.opinion() is None:
            raise ValueError('Please initialize the user to compute a reward.')
        return self.get_parameters().reward(opinion=self.opinion(), recommendation=recommendation)

    def update_state(self, recommendation: Recommendation) -> Reward:
        if self.opinion() is None:
            raise ValueError('Initialize state before updating it.')
        reward = self.reward(recommendation)
        x_new = self.get_parameters().weight_prejudice * self.get_parameters().prejudice \
                + self.get_parameters().weight_current_opinion * self.opinion() \
                + self.get_parameters().weight_recommendation * recommendation
        self._x = Opinion(x_new)
        if self.save_history:
            self.trajectory.append(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION],
                                   items=[x_new, reward, recommendation])
        return reward

    def __eq__(self, other) -> bool:
        return self.get_parameters() == other.get_parameters()

    def plot(self, save: bool = False, show: bool = True, name: str = None, folder: str = None) -> None:
        if not self.save_history:
            return
        self.trajectory.plot(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION],
                             show=show,
                             save=save,
                             name=name,
                             folder=folder)

    def save_trajectory_to_file(self, name: str = None, folder: str = None) -> None:
        self.trajectory.save_to_file(name=name, folder=folder)


class Population(OpinionDynamicsEntity):
    def __init__(self,
                 parameters: ParametersPopulation,
                 initial_state: Opinion or SamplerOpinion or None = None,
                 initialize_with_prejudice: bool = False,
                 save_history: bool = True) -> None:
        super().__init__(initial_state=initial_state,
                         save_history=save_history)
        self._n_agents = parameters.n_agents()
        self._users = []
        self._identical_same_prejudice = parameters.identical_same_prejudice()
        self._identical_different_prejudice = parameters.identical_different_prejudice()
        self._parameters = parameters
        self.trajectory = Trajectory([KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD, KEY_AVERAGE_OPINION,
                                      KEY_STD_OPINION, KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION])
        # this will initialize the users again -- thus, the initialization above does not matter
        self.initialize(initial_state=self.initial_state(),
                        initialize_with_prejudice=initialize_with_prejudice)

    def n_agents(self) -> int:
        return self._n_agents

    def identical_same_prejudice(self) -> bool:
        return self._identical_same_prejudice

    def identical_different_prejudice(self) -> bool:
        return self._identical_different_prejudice

    def get_user(self, idx: int) -> User:
        return User(parameters=self.parameters_user(item=idx),
                    initial_state=self.opinions()[idx],
                    save_history=False)

    def parameters(self) -> ParametersPopulation:
        return self._parameters

    def parameters_user(self, item: int) -> ParametersUser:
        return self.parameters()[item]

    def add_user(self, user: User) -> None:
        raise ValueError('Not implemented.')

    def initialize(self,
                   initial_state: Opinion or SamplerOpinion or None = None,
                   initialize_with_prejudice: bool = False) -> None:
        if initialize_with_prejudice and initial_state is not None:
            raise ValueError('Either initialize with prejudice or input a given state.')
        elif initialize_with_prejudice:
            if self.identical_same_prejudice():
                self._x = self.parameters().prejudice * np.ones(self.n_agents())
            else:
                self._x = self.parameters().prejudice
        elif isinstance(initial_state, SamplerOpinion):
            self._x = initial_state.sample(number=self.n_agents())  # sample one
        elif isinstance(initial_state, OpinionType):
            self._x = initial_state
        elif initial_state is None:
            if isinstance(self.initial_state(), SamplerOpinion):
                self._x = self.initial_state().sample(number=self.n_agents())
            else:
                self._x = self.initial_state()
        else:
            raise ValueError('Unknown input, received ' + type(initial_state) + '.')
        self._last_initial_state = self.opinions()
        if self.save_history:
            self.trajectory.reset()
            self.trajectory.append(keys=[KEY_OPINION, KEY_AVERAGE_OPINION, KEY_STD_OPINION],
                                   items=[self.opinions(), self.average_opinion(), self.std_opinion()])

    def update_state(self, recommendation: Recommendation) -> Reward:
        if self.identical_same_prejudice() or self.identical_different_prejudice():
            reward = self.parameters().reward(opinion=self.opinions(), recommendation=recommendation)
            x_new = self.parameters().weight_prejudice * self.parameters().prejudice \
                    + self.parameters().weight_current_opinion * self.opinions() \
                    + self.parameters().weight_recommendation * recommendation
        else:
            reward = [self.parameters_user(i).reward(opinion=self.opinions()[i],
                                                     recommendation=r) for i, r in enumerate(recommendation)]
            reward = Reward(np.asarray(reward))
            x_new = np.multiply(self.parameters().weight_prejudice, self.parameters().prejudice) \
                    + np.multiply(self.parameters().weight_current_opinion, self.opinions()) \
                    + np.multiply(self.parameters().weight_recommendation, recommendation)
        self._x = Opinion(x_new)
        if self.save_history:
            self.trajectory.append(keys=[KEY_OPINION, KEY_REWARD, KEY_RECOMMENDATION, KEY_AVERAGE_OPINION,
                                         KEY_STD_OPINION, KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION],
                                   items=[self.opinions(), reward, recommendation,
                                          self.average_opinion(), self.std_opinion(),
                                          reward.mean(), recommendation.mean()])
        return reward

    def opinions(self) -> Opinion:
        return self._x

    def average_opinion(self) -> float:
        return self.opinions().mean()

    def variance_opinion(self) -> float:
        return self.opinions().var()

    def std_opinion(self) -> float:
        return self.opinions().std()

    def __add__(self, other):
        if isinstance(other, Population):
            return Populations(populations=[self, other])
        elif isinstance(other, Populations):
            other.add_population(population=self)
            return other
        else:
            raise ValueError('Unknown input type.')

    def __eq__(self, other) -> bool:
        if isinstance(other, Population):
            if self.n_agents() != other.n_agents():
                return False
            if self.parameters() != other.parameters():
                return False
            return True
        else:
            raise ValueError("You can only compare two populations.")

    def save_trajectory_to_file(self, name: str = None, folder: str = None) -> None:
        self.trajectory.save_to_file(name=name, folder=folder)

    def plot(self, save: bool = False, show: bool = True, name: str = None, folder: str = None,
             intermediate: float = 0.5) -> tuple:
        if not self.save_history:
            raise ValueError('Cannot produce plot, no history saved.')
        # average opinion, std, etc
        self.trajectory.plot(axis=None,
                             keys=[KEY_AVERAGE_OPINION, KEY_STD_OPINION, KEY_AVERAGE_REWARD,
                                   KEY_AVERAGE_RECOMMENDATION],
                             color='blue',
                             show=show,
                             save=save,
                             name=name,
                             folder=folder)
        # plot Jules
        plot_opinion_shift(axis=None,
                           axis_hist=None,
                           x_start=self.trajectory[KEY_OPINION][0],
                           x_end=self.trajectory[KEY_OPINION][-1],
                           color='blue',
                           show=show,
                           save=save,
                           name=name,
                           folder=folder)
        # histograms
        length = self.trajectory.get_number_entries_item(KEY_OPINION)
        plot_opinions_time(axis=None,
                           x=[self.trajectory[KEY_OPINION][0],
                              self.trajectory[KEY_OPINION][int(intermediate * length)],
                              self.trajectory[KEY_OPINION][-1]],
                           color='blue',
                           labels=['Initial', 'Intermediate', 'Final'],
                           show=show,
                           save=save,
                           name=name + 'opinions_time',
                           folder=folder)
        # Sankey
        sankey_plot_data = plot_sankey_single_population(x=(self.trajectory[KEY_RECOMMENDATION][0],
                                                            self.trajectory[KEY_OPINION][-1]))
        return sankey_plot_data


class Populations(OpinionDynamicsEntity):
    def __init__(self,
                 populations: List[Population] or None  = None,
                 parameters: List[ParametersPopulation] or  None = None,
                 initial_state: List[Opinion] or List[SamplerOpinion] or None = None,
                 initialize_with_prejudice: bool = False,
                 save_history: bool = True) -> None:
        super().__init__(initial_state=initial_state,
                         save_history=save_history)
        if populations is not None and isinstance(populations, List):  # populations already in the right format
            self._populations = populations
        elif parameters is not None and isinstance(parameters, list):  # populations first to create
            self._populations = []
            if initial_state is None:
                initial_state = [None] * len(parameters)
            for i, p in enumerate(parameters):
                self.add_population(parameters=p,
                                    initial_state=initial_state[i],
                                    initialize_with_prejudice=initialize_with_prejudice,
                                    save_history=save_history)

    def add_population(self,
                       population: Population or None = None,
                       parameters: ParametersPopulation or None = None,
                       initial_state: Opinion or SamplerOpinion or None = None,
                       initialize_with_prejudice:  bool = False,
                       save_history: bool = True):
        if population is not None:
            assert isinstance(population, Population),'Please check type of population.'
            self._populations.append(population)
        elif parameters is not None:
            self._populations.append(Population(parameters=parameters,
                                                initial_state=initial_state,
                                                initialize_with_prejudice=initialize_with_prejudice,
                                                save_history=save_history))
        else:
            raise ValueError('Either input a population or its parameters.')

    def __getitem__(self, idx: int):
        if isinstance(idx, int):
            return self.populations()[idx]
        else:
            raise ValueError('Unknown input type.')

    def __call__(self, *args, **kwargs):
        if isinstance(args, int):
            return self.populations()[args]
        else:
            raise ValueError('Unknown input type.')

    def n_populations(self) -> int:
        return len(self.populations())

    def n_agents(self) -> List[int]:
        return [p.n_agents() for p in self.populations()]

    def populations(self):
        return self._populations

    def initialize(self,
                   initial_state: List[Opinion] or List[SamplerOpinion] or None = None,
                   initialize_with_prejudice: List[bool] or bool or None = None) -> None:
        if initial_state is None:
            initial_state = [None] * self.n_populations()
        if initialize_with_prejudice is None:
            initialize_with_prejudice = [None] * self.n_populations()
        if isinstance(initialize_with_prejudice, bool):
            initialize_with_prejudice = [initialize_with_prejudice] * self.n_populations()
        assert len(initialize_with_prejudice) == self.n_populations(),'Check size of initialize_with_prejudice.'
        assert len(initial_state) == self.n_populations(),'Check size of initial_state.'
        for i in range(self.n_populations()):
            self._populations[i].initialize(initial_state=initial_state[i],
                                            initialize_with_prejudice=initialize_with_prejudice[i])

    def update_state(self, recommendation: List[Recommendation]) -> List[Reward]:
        reward = []
        for i in range(self.n_populations()):
            reward.append(self._populations[i].update_state(recommendation=recommendation[i]))
        return reward

    def parameters_populations(self) -> List[ParametersPopulation]:
        return [p.parameters for p in self.populations()]

    def opinions(self) -> List[Opinion]:
        return [p.opinions() for p in self.populations()]

    def average_opinion(self) -> List[Opinion]:
        return [p.average_opinion() for p in self.populations()]

    def variance_opinion(self) -> List[Opinion]:
        return [p.variance_opinion() for p in self.populations()]

    def std_opinion(self) -> List[Opinion]:
        return [p.std_opinion() for p in self.populations()]

    def __eq__(self, other) -> bool:
        if self.n_populations() == other.n_populations():
            equal = True
            for i, p in enumerate(self.populations()):
                equal = equal and p == other(i)
            return equal
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Population):
            self.add_population(population=other)
        elif isinstance(other, Populations):
            for p in other.populations():
                self.add_population(population=p)
        else:
            raise ValueError('Unknown input type.')
        return self

    def plot(self, save: bool = False, show: bool = True, name: str = None, folder: str = None,
             intermediate: float = 0.5) -> tuple:
        if not self.save_history:
            raise ValueError('Cannot produce plot, no history saved.')
        _, ax_aggregated_stuff = plt.subplots(nrows=4, ncols=1)
        _, ax_opinion_shift = plt.subplots(nrows=1, ncols=1)
        divider = make_axes_locatable(ax_opinion_shift)
        ax_opinion_shift_hist = (divider.append_axes("top", 1, pad=0.15, sharex=ax_opinion_shift),
                                 divider.append_axes("right", 1, pad=0.2, sharey=ax_opinion_shift))
        _, ax_opinion_time = plt.subplots(nrows=self.n_populations(), ncols=1)
        colors = ['blue', 'red', 'green', 'yellow', 'magenta']
        for i, p in enumerate(self.populations()):
            p.trajectory.plot(axis=ax_aggregated_stuff,
                              keys=[KEY_AVERAGE_OPINION, KEY_STD_OPINION,
                                    KEY_AVERAGE_REWARD, KEY_AVERAGE_RECOMMENDATION],
                              color=colors[i],
                              show=False,
                              save=save,
                              name=name + '_trajectory',
                              folder=folder)
            # plot Jules
            plot_opinion_shift(axis=ax_opinion_shift,
                               axis_hist=ax_opinion_shift_hist,
                               x_start=p.trajectory[KEY_OPINION][0],
                               x_end=p.trajectory[KEY_OPINION][-1],
                               color=colors[i],
                               show=False,
                               save=save,
                               name=name + '_opinions',
                               folder=folder)
            # histograms
            length = p.trajectory.get_number_entries_item(KEY_OPINION)
            plot_opinions_time(axis=ax_opinion_time[i],
                               x=[p.trajectory[KEY_OPINION][0],
                                  p.trajectory[KEY_OPINION][int(intermediate * length)],
                                  p.trajectory[KEY_OPINION][-1]],
                               color=colors[i],
                               labels=['Initial', 'Intermediate', 'Final'],
                               show=False,
                               save=save,
                               name=name + '_opinions_time',
                               folder=folder)
        # Sankey
        sankey_plot_data = plot_sankey_multiple_populations(x=([(p.trajectory[KEY_RECOMMENDATION][0],
                                                                 p.trajectory[KEY_OPINION][-1]) for p in
                                                                self.populations()]))
        if show:
            plt.show()
        return sankey_plot_data

    def save_trajectory_to_file(self, name: str = None, folder: str = None) -> None:
        for i, p in enumerate(self.populations()):
            p.trajectory.save_to_file(name=name + 'population_' + str(i + 1), folder=folder)
