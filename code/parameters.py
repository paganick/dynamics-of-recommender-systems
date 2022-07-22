import numpy as np
from abc import ABC, abstractmethod
from typing import List
from utils import Opinion
from rewards import RewardFunction, load_reward_function
import copy


class Parameters(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def save(self) -> dict:
        pass


class ParametersUser(Parameters):
    def __init__(self,
                 prejudice: Opinion,
                 weight_prejudice: float,
                 weight_current_opinion: float,
                 weight_recommendation: float,
                 reward: RewardFunction) -> None:
        super().__init__()
        self.prejudice = prejudice
        self.weight_prejudice = weight_prejudice
        self.weight_current_opinion = weight_current_opinion
        self.weight_recommendation = weight_recommendation
        assert self.check(), 'Invalid input parameters, they should sum to 1.'
        self.reward = reward

    def check(self) -> bool:
        return self.weight_prejudice + self.weight_current_opinion + self.weight_recommendation == 1

    def save(self) -> dict:
        return {'prejudice': self.prejudice,
                'weight_prejudice': self.weight_prejudice,
                'weight_current_opinion': self.weight_prejudice,
                'weight_recommendation': self.weight_recommendation,
                'reward': self.reward.save()}


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
            self.identical = False
            self.n_agents = len(parameters)
            if repeat is not None and repeat != 1:
                raise ValueError('Currently not supported, repeat only works if one parameters is inputed.')
        elif isinstance(parameters, ParametersUser):
            self.identical = True
            self.n_agents = repeat
        else:
            raise ValueError('Unknown input type.')
        self.parameters = parameters

    def get_parameters_user(self, idx_user: int) -> ParametersUser:
        if self.identical:
            return self.parameters
        else:
            return self.parameters[idx_user]

    def save(self) -> dict:
        if self.identical:
            out = {'n_agents': self.n_agents,
                   'parameter_users': self.parameters}
        else:
            out = {}
            for i, p in enumerate(self.parameters):
                out['user_' + str(i)] = p.save()
        return out


def load_parameters_population(parameters: dict) -> ParametersPopulation:
    pass #TODO implement
