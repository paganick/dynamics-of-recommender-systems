import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from modules.utils import ListRecommendation, Recommendation
from modules.utils import KEY_SAMPLER_RECOMMENDATION_TYPE
from modules.utils import KEY_SAMPLER_RECOMMENDATION_TYPE_UNIFORM, KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW, KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH
from modules.utils import KEY_SAMPLER_RECOMMENDATION_TYPE_GAUSSIAN, KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_MEAN, KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_STD


class SamplerRecommendation(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod  # TODO: implement this correctly
    def set_seed(seed: int):
        raise ValueError('This is not implemented yet.')

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

    def plot(self) -> None:  # TODO: improve
        samples = self.sample(number=1000)
        plt.hist(samples)


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
        return {KEY_SAMPLER_RECOMMENDATION_TYPE: KEY_SAMPLER_RECOMMENDATION_TYPE_UNIFORM,
                KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW: self._low,
                KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH: self._high}

    def get_expected_value(self) -> float:
        return 0.5*(self._low + self._high)

    def get_variance(self) -> float:
        return (self._high-self._low)**2/12.0

    def get_standard_deviation(self) -> float:
        return np.sqrt(self.get_variance())


class GaussianSamplerRecommendation(SamplerRecommendation):
    def __init__(self,
                 mean: float,
                 std: float) -> None:
        super().__init__()
        self._mean = mean
        self._std = std

    def sample(self,
               number: int) -> ListRecommendation:
        return ListRecommendation(self._mean + self._std*np.random.randn(number))

    def save(self) -> dict:
        return {KEY_SAMPLER_RECOMMENDATION_TYPE: KEY_SAMPLER_RECOMMENDATION_TYPE_GAUSSIAN,
                KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_MEAN: self._mean,
                KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_STD: self._std}

    def get_expected_value(self) -> float:
        return self._mean

    def get_variance(self) -> float:
        return self.get_standard_deviation()**2

    def get_standard_deviation(self) -> float:
        return self._std


class BimodalSamplerRecommendation(SamplerRecommendation):
    def __init__(self,
                 mean: list,
                 std: list) -> None:
        super().__init__()
        self._mean = np.asarray(mean)
        self._std = np.asarray(std)

    def sample(self,
               number: int) -> ListRecommendation:
        which = np.random.rand(number) <= 0.5
        r1 = ListRecommendation(self._mean[0] + self._std[0]*np.random.randn(number))
        r2 = ListRecommendation(self._mean[1] + self._std[1]*np.random.randn(number))
        return r1*which + r2*(1.0-which)

    def save(self) -> dict:
        raise ValueError('Not implemented.')

    def get_expected_value(self) -> np.ndarray:
        return self._mean

    def get_variance(self) -> np.ndarray:
        return self.get_standard_deviation()**2

    def get_standard_deviation(self) -> np.ndarray:
        return self._std


def load_import_recommendation_sampler(parameters:dict) -> SamplerRecommendation:
    if not isinstance(parameters, dict) or KEY_SAMPLER_RECOMMENDATION_TYPE not in parameters:
        raise ValueError('Unknown input type.')
    if parameters['KEY_SAMPLER_RECOMMENDATION_TYPE'] ==  KEY_SAMPLER_RECOMMENDATION_TYPE_UNIFORM:
        return UniformSamplerRecommendation(low=parameters[KEY_SAMPLER_RECOMMENDATION_UNIFORM_LOW],
                                            high=parameters[KEY_SAMPLER_RECOMMENDATION_UNIFORM_HIGH])
    elif parameters['KEY_SAMPLER_RECOMMENDATION_TYPE'] ==  KEY_SAMPLER_RECOMMENDATION_TYPE_GAUSSIAN:
        return GaussianSamplerRecommendation(mean=parameters[KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_MEAN],
                                             std=parameters[KEY_SAMPLER_RECOMMENDATION_GAUSSIAN_STD])
    else:
        raise ValueError('Unknown distribution.')
