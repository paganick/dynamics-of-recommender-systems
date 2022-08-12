import numpy as np
from abc import ABC, abstractmethod
from modules.utils import Reward, KEY_SQUARED_EXPONENTIAL_REWARD, KEY_EXPONENTIAL_REWARD, KEY_REWARD_TYPE, KEY_REWARD_DECAY_PARAMETER


class RewardFunction(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def save(self) -> dict:
        pass

    @abstractmethod
    def __call__(self, distance: float) -> Reward:
        pass

    @staticmethod
    def check_distance(distance: float) -> None:
        assert distance >= 0, 'The distance should be non-negative.'


class RewardFunctionSquaredExponential(RewardFunction):
    def __init__(self,
                 decay_parameter: float) -> None:
        super().__init__()
        assert decay_parameter > 0, 'The decay parameter should be non-negative.'
        self.decay_parameter = decay_parameter

    def __call__(self, distance: float):
        self.check_distance(distance)
        return Reward(np.exp(-self.decay_parameter*(distance**2)))

    def save(self) -> dict:
        return {KEY_REWARD_TYPE: KEY_SQUARED_EXPONENTIAL_REWARD,
                KEY_REWARD_DECAY_PARAMETER: self.decay_parameter}


class RewardFunctionExponential(RewardFunction):
    def __init__(self,
                 decay_parameter: float) -> None:
        super().__init__()
        assert decay_parameter > 0, 'The decay parameter should be non-negative.'
        self.decay_parameter = decay_parameter

    def __call__(self, distance: float):
        self.check_distance(distance)
        return Reward(np.exp(-self.decay_parameter*distance))

    def save(self) -> dict:
        return {KEY_REWARD_TYPE: KEY_EXPONENTIAL_REWARD,
                KEY_REWARD_DECAY_PARAMETER: self.decay_parameter}


def load_reward_function(parameters: dict) -> RewardFunction:
    if KEY_REWARD_TYPE not in parameters.keys():
        raise ValueError('Wrong input data.')
    if parameters[KEY_REWARD_TYPE] == KEY_EXPONENTIAL_REWARD:
        return RewardFunctionExponential(parameters[KEY_REWARD_DECAY_PARAMETER])
    elif parameters[KEY_REWARD_TYPE] == KEY_SQUARED_EXPONENTIAL_REWARD:
        return RewardFunctionSquaredExponential(parameters[KEY_REWARD_DECAY_PARAMETER])
    else:
        raise ValueError('Unknown type of reward function')