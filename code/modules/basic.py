from abc import ABC
import numpy as np
from typing import NewType


OpinionType = np.ndarray
Opinion = NewType('Opinion', OpinionType)  # TODO: make this work

RewardType = np.ndarray
Reward = NewType('Reward', RewardType)

RecommendationType = np.ndarray
Recommendation = NewType('Recommendation', RecommendationType)

"""
class BasicObject(ABC):
    def __init__(self, value: float or np.ndarray or list or None):
        if isinstance(value, float):
            self._value = np.array([value])
        elif isinstance(value, list):
            self._value = np.asarray(value)
        elif isinstance(value, np.ndarray):
            self._value = value
        else:
            raise ValueError('Unknown input type, given ' + type(value) + '.')

    def value(self) -> np.ndarray:
        return self._value

    def as_array(self) -> np.ndarray:
        return np.asarray(self.value()).reshape(-1)

    def as_float(self) -> float:
        if self.size() == 1:
            return float(self.value())
        else:
            raise ValueError('The output should be one-dimensional to become a float.')

    def set(self, idx: np.ndarray or int, value) -> None:
        if isinstance(idx, int) and isinstance(value, (float, int)):
            self._value[idx] = value
        elif isinstance(idx, np.ndarray) and isinstance(value, np.ndarray) and idx.size == value.size:
            self._value[idx] = value
        elif isinstance(idx, np.ndarray) and isinstance(value, BasicObject) and idx.size == value.size():
            self._value[idx] = value.value()
        else:
            raise ValueError('Unknown input type.')

    def size(self) -> int:
        return self.value().size

    def mean(self) -> float:
        return float(np.mean(self.value()))

    def variance(self) -> float:
        return float(np.var(self.value()))

    def std(self) -> float:
        return float(np.std(self.value()))

    def __getitem__(self, idx: int or np.ndarray):
        return self.value()[idx]

    def is_nan(self) -> bool:
        return np.any(np.isnan(self.value()))

    def __eq__(self, other):
        return np.all(self.value() == other.value())

    def __gt__(self, other):
        return np.all(self.value() > other.value())

    def __ge__(self, other):
        return np.all(self.value() >= other.value())

    def find_idx_is_larger_than(self, other):
        return np.where(self.value() > other.value())[0]

    def __add__(self, other):
        return self.value() + other.value()

    def __sub__(self, other):
        return self.value() - other.value()

    def __mul__(self, other: float):
        if isinstance(other, float):
            return other*self.value()
        else:
            raise ValueError('Unknown input type.')

    def __rmul__(self, other: float):
        return self.__mul__(other)


class Opinion(BasicObject):
    def __init__(self, value: float or np.ndarray):
        super().__init__(value=value)

    def __getitem__(self, idx: int or np.ndarray):
        return Opinion(super().__getitem__(idx))

    def __add__(self, other):
        if isinstance(other, (Opinion, Recommendation)):
            return Opinion(super().__add__(other))
        else:
            raise ValueError('Can only add opinion and opinion, or opinion and recommendation.')

    def __mul__(self, other: float):
        return Opinion(super().__mul__(other))


class Reward(BasicObject):
    def __init__(self, value: float or np.ndarray):
        super().__init__(value=value)

    def __getitem__(self, idx: int or np.ndarray):
        return Reward(super().__getitem__(idx))

    def __add__(self, other):
        if isinstance(other, Reward):
            return Reward(super().__add__(other))
        else:
            raise ValueError('Can only add reward and reward.')

    def __mul__(self, other: float):
        return Reward(super().__mul__(other))


class Recommendation(BasicObject):
    def __init__(self, value: float or np.ndarray):
        super().__init__(value=value)

    def __getitem__(self, idx: int or np.ndarray):
        return Recommendation(super().__getitem__(idx))

    def __add__(self, other):
        if isinstance(other, Recommendation):
            return Recommendation(super().__add__(other))
        else:
            raise ValueError('Can only add recommendation and recommendation.')

    def __sub__(self, other):
        if isinstance(other, Recommendation):
            return Recommendation(super().__sub__(other))
        else:
            raise ValueError('Can only subtract recommendation and recommendation.')

    def __mul__(self, other: float):
        return Recommendation(super().__mul__(other))
"""