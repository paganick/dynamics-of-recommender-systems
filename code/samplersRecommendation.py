import numpy as np
from abc import ABC, abstractmethod
from utils import ListRecommendation, Recommendation, convert_list_to_recommendation
from utils import KEY_SAMPLER_RECOMMENDATION_TYPE, KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW, KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH


class SamplerRecommendation(ABC):
    def __init__(self) -> None:
        pass

    def sample(self,
               number: int) -> ListRecommendation or Recommendation:
        pass

    def save(self) -> dict:
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
