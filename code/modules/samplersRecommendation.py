import numpy as np
from abc import ABC, abstractmethod
from modules.utils import ListRecommendation, Recommendation
from modules.utils import KEY_SAMPLER_RECOMMENDATION_TYPE, KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW, KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH


class SamplerRecommendation(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod #TODO: implement this correctly
    def set_seed(seed: int):
        if isinstance(seed, int):
            np.random.seed(seed)
        else:
            raise ValueError('Unknown seed, please input an integer.')

    @abstractmethod
    def sample(self,
               number: int) -> ListRecommendation or Recommendation:
        pass

    @abstractmethod
    def save(self) -> dict:
        pass

    @abstractmethod
    def get_expected_value(self) -> float:
        pass

    @abstractmethod
    def get_variance(self) -> float:
        pass

    @abstractmethod
    def get_standard_deviation(self) -> float:
        pass


class UniformSamplerRecommendation(SamplerRecommendation):
    def __init__(self,
                 low: float,
                 high: float) -> None:
        super().__init__()
        self._low = low
        self._high = high

    def sample(self,
               number: int) -> ListRecommendation:
        return ListRecommendation(np.random.uniform(low=self._low,
                                                    high=self._high,
                                                    size=number))

    def save(self) -> dict:
        return {KEY_SAMPLER_RECOMMENDATION_TYPE: 'uniform',
                KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW: self._low,
                KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH: self._high}

    def get_expected_value(self) -> float:
        return 0.5*(self._low + self._high)

    def get_variance(self) -> float:
        return (self._high-self._low)**2/12.0

    def get_standard_deviation(self) -> float:
        return np.sqrt(self.get_variance())
