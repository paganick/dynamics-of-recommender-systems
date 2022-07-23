from abc import ABC, abstractmethod
import numpy as np
from typing import List
from utils import Opinion, KEY_PARAMETERS_USERS, KEY_N_AGENTS, KEY_USER
from rewards import RewardFunction, load_reward_function


class Parameters(ABC):
    def __init__(self) -> None:
        pass

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
        x = 0.0  # compare on a grid of points
        while ok_5 and x <= 2.0:
            ok_5 = ok_5 and np.abs(self.reward(x) - other.reward(x)) < 1e-6
            x += 1e-2
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
            self.identical = False
            self.n_agents = len(parameters)
            if repeat is not None and repeat != 1:
                raise ValueError('Currently not supported, repeat only works if one parameters is he input.')
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
            out = {KEY_N_AGENTS: self.n_agents,
                   KEY_PARAMETERS_USERS: self.parameters}
        else:
            out = {}
            for i, p in enumerate(self.parameters):
                out[KEY_USER + '_' + str(i)] = p.save()
        return out

    def __eq__(self, other) -> bool:
        if self.n_agents != other.n_agents:
            return False
        if self.identical and other.identical:
            return self.get_parameters_user(0) == other.get_parameters_user(0)
        else:
            for i in range(self.n_agents):
                if self.get_parameters_user(i) != other.get_parameters_user(i):
                    return False
            return True


def load_parameters_population(parameters: dict) -> ParametersPopulation:
    if KEY_N_AGENTS in parameters:
        return ParametersPopulation(parameters=parameters[KEY_PARAMETERS_USERS],
                                    repeat=parameters[KEY_N_AGENTS])
    elif KEY_USER + '_1' in parameters:
        parameters_users_input = []
        for key, par in parameters.items():
            if KEY_USER in key:
                parameters_users_input.append(load_parameters_user(par))
        return ParametersPopulation(parameters=parameters_users_input)
    else:
        ValueError('Unknown input type. ')
