from abc import ABC, abstractmethod
import numpy as np
from typing import List
from modules.basic import Opinion, Recommendation
from modules.utils import KEY_PARAMETERS_USERS, KEY_N_AGENTS, KEY_USER
from modules.rewardsFunctions import RewardFunction, load_reward_function
from modules.samplers import SamplerOpinion


class Parameters(ABC):
    def __init__(self) -> None:
        self.prejudice = None
        self.weight_prejudice = None
        self.weight_current_opinion = None
        self.weight_recommendation = None
        self.reward = None

    @abstractmethod
    def save(self) -> dict:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    def __str__(self) -> None:
        print(self.save())


class ParametersUser(Parameters):
    def __init__(self,
                 prejudice: Opinion or float,  # TODO: implement sampler as prejudice
                 weight_prejudice: float,
                 weight_current_opinion: float,
                 weight_recommendation: float,
                 reward: RewardFunction) -> None:
        super().__init__()
        if isinstance(prejudice, float):
            prejudice = Opinion(np.array([prejudice]))
        self.prejudice = prejudice
        self.weight_prejudice = weight_prejudice
        self.weight_current_opinion = weight_current_opinion
        self.weight_recommendation = weight_recommendation
        assert self.check(), 'Invalid input parameters, they should sum to 1.'
        self.reward = reward

    def check(self) -> bool:
        sum_parameters = self.weight_prejudice + self.weight_current_opinion + self.weight_recommendation
        return np.abs(sum_parameters - 1.0) <= 1e-6

    def save(self) -> dict:
        return {'prejudice': self.prejudice,
                'weight_prejudice': self.weight_prejudice,
                'weight_current_opinion': self.weight_current_opinion,
                'weight_recommendation': self.weight_recommendation,
                'reward': self.reward.save()}

    def __eq__(self, other) -> bool:
        ok_1 = self.prejudice == other.prejudice
        ok_2 = self.weight_prejudice == other.weight_prejudice
        ok_3 = self.weight_current_opinion == other.weight_current_opinion
        ok_4 = self.weight_recommendation == other.weight_recommendation
        ok_5 = True
        n_points = 20
        o, r = np.meshgrid(np.linspace(-2.0, 2.0, n_points), np.linspace(-2.0, 2.0, n_points))
        for i in range(n_points):
            if ok_5:
                for j in range(n_points):
                    oij, rij = Opinion(o[i,j]), Recommendation(r[i,j])
                    ok_5 = ok_5 and np.abs(self.reward(oij, rij) - other.reward(oij, rij)) < 1e-6
            else:
                break
        return ok_1 and ok_2 and ok_3 and ok_4 and ok_5


def load_parameters_user(parameters: dict) -> ParametersUser:
    return ParametersUser(prejudice=Opinion(parameters['prejudice']),
                          weight_prejudice=parameters['weight_prejudice'],
                          weight_current_opinion=parameters['weight_current_opinion'],
                          weight_recommendation=parameters['weight_recommendation'],
                          reward=load_reward_function(parameters['reward']))


class ParametersPopulation(Parameters):
    def __init__(self,
                 parameters: ParametersUser or List[ParametersUser],
                 repeat: int or None = None) -> None:
        super().__init__()
        if isinstance(parameters, list):
            self._identical = False
            self._n_agents = len(parameters)
            self.prejudice = np.asarray([p.prejudice for p in parameters])
            self.weight_prejudice = np.asarray([p.weight_prejudice for p in parameters])
            self.weight_current_opinion = np.asarray([p.weight_current_opinion for p in parameters])
            self.weight_recommendation = np.asarray([p.weight_recommendation for p in parameters])
            self.reward = [p.reward for p in parameters]
            if repeat is not None and repeat != 1:
                raise ValueError('Currently not supported, repeat only works if one parameter is he input.')
        elif isinstance(parameters, ParametersUser):
            self._identical = True
            self._n_agents = repeat
            self.prejudice = parameters.prejudice
            self.weight_prejudice = parameters.weight_prejudice
            self.weight_current_opinion = parameters.weight_current_opinion
            self.weight_recommendation = parameters.weight_recommendation
            self.reward = parameters.reward
        else:
            raise ValueError('Unknown input type.')

    def identical(self) -> bool:
        return self._identical

    def n_agents(self) -> int:
        return self._n_agents

    def __getitem__(self, item: int) -> ParametersUser:
        if self.identical():
            return ParametersUser(prejudice=self.prejudice,
                                  weight_prejudice=self.weight_prejudice,
                                  weight_current_opinion=self.weight_current_opinion,
                                  weight_recommendation=self.weight_recommendation,
                                  reward=self.reward)
        else:
            return ParametersUser(prejudice=self.prejudice[item],
                                  weight_prejudice=self.weight_prejudice[item],
                                  weight_current_opinion=self.weight_current_opinion[item],
                                  weight_recommendation=self.weight_recommendation[item],
                                  reward=self.reward[item])

    def save(self) -> dict:
        if self.identical():
            out = {KEY_N_AGENTS: self.n_agents(),
                   KEY_PARAMETERS_USERS: self.__getitem__(item=0).save()}  # simply save one of the users
        else:
            out = {}
            for i in range(self.n_agents()):
                out[KEY_USER + '_' + str(i)] = self.__getitem__(item=i).save()
        return out

    def __eq__(self, other) -> bool:
        if self.n_agents() != other.n_agents():
            return False
        if self.identical() and other.identical():
            return self.__getitem__(0) == other.__getitem__(0)
        else:
            for i in range(self.n_agents()):
                if self.__getitem__(i) != other.__getitem__(i):
                    return False
            return True


def load_parameters_population(parameters: dict) -> ParametersPopulation:
    if KEY_N_AGENTS in parameters:
        return ParametersPopulation(parameters=load_parameters_user(parameters[KEY_PARAMETERS_USERS]),
                                    repeat=parameters[KEY_N_AGENTS])
    elif KEY_USER + '_1' in parameters:
        parameters_users_input = []
        for key, par in parameters.items():
            if KEY_USER in key:
                parameters_users_input.append(load_parameters_user(par))
        return ParametersPopulation(parameters=parameters_users_input)
    else:
        ValueError('Unknown input type. ')
