import numpy as np
import scipy
from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt
from modules.basic import Opinion, Recommendation
from modules.utils import KEY_SAMPLER_TYPE, KEY_SAMPLER_OBJECT
from modules.utils import KEY_SAMPLER_OBJECT_RECOMMENDATION, KEY_SAMPLER_OBJECT_OPINION
from modules.utils import KEY_SAMPLER_TYPE_UNIFORM, KEY_SAMPLER_UNIFORM_LOW, KEY_SAMPLER_UNIFORM_HIGH
from modules.utils import KEY_SAMPLER_TYPE_GAUSSIAN, KEY_SAMPLER_GAUSSIAN_MEAN, KEY_SAMPLER_GAUSSIAN_STD
from modules.utils import KEY_SAMPLER_TYPE_MIXTURE_GAUSSIAN, KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN, KEY_SAMPLER_MIXTURE_GAUSSIAN_STD


class Sampler(ABC):  # TODO: distinguish abstract sampler, basic sampler, and mixture sampler
    def __init__(self) -> None:
        self._distribution = None

    def sample(self,
               number: int,
               seed: int or None = None) -> np.ndarray:
        return self.distribution().rvs(size=number, random_state=seed)

    @abstractmethod
    def save(self) -> dict:
        pass

    def distribution(self) -> scipy.stats.rv_continuous or List[scipy.stats.rv_continuous]:
        return self._distribution

    def expected_value(self) -> float:
        return float(self.distribution().mean())

    def variance(self) -> float:
        return float(self.distribution().var())

    def standard_deviation(self) -> float:
        return float(self.distribution().std())

    def support(self) -> tuple:
        return self.distribution().support()

    def plot(self, show: bool = True, color: str = 'blue') -> None:
        samples = self.sample(number=10000)
        plt.hist(x=samples, bins=50, color=color)
        if show:
            plt.show()


class SamplerRecommendation(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self,
               number: int,
               seed: int or None = None) -> Recommendation:
        return Recommendation(super().sample(number=number, seed=seed))

    def save(self) -> dict:
        out = super().save()
        out[KEY_SAMPLER_OBJECT] = KEY_SAMPLER_OBJECT_RECOMMENDATION
        return out

    def plot(self, show: bool = True, color: str = 'blue') -> None:
        super().plot(show=False)
        plt.xlabel('Recommendation')
        if show:
            plt.show()


class SamplerOpinion(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self,
               number: int,
               seed: int or None = None) -> Opinion:
        return Opinion(super().sample(number=number, seed=seed))

    def save(self) -> dict:
        out = super().save()
        out[KEY_SAMPLER_OBJECT] = KEY_SAMPLER_OBJECT_OPINION
        return out

    def plot(self, show: bool = True, color: str = 'blue') -> None:
        super().plot(show=False, color=color)
        plt.xlabel('Opinion')
        if show:
            plt.show()


class UniformSampler(Sampler):
    def __init__(self,
                 low: float,
                 high: float) -> None:
        super().__init__()
        self.check_parameters(low=low, high=high)
        self._distribution = scipy.stats.uniform(loc=low, scale=high-low)

    @staticmethod
    def check_parameters(low: float, high: float) -> None:
        if isinstance(low, np.ndarray) and low.size == 1:
            low = low[0]
        if isinstance(high, np.ndarray) and high.size == 1:
            high = high[0]
        assert isinstance(low, float), 'low must be a float.'
        assert isinstance(high, float), 'high must be a float.'
        assert high >= low, 'high must be larger than low.'

    def save(self) -> dict:
        return {KEY_SAMPLER_TYPE: KEY_SAMPLER_TYPE_UNIFORM,
                KEY_SAMPLER_UNIFORM_LOW: self.support()[0],
                KEY_SAMPLER_UNIFORM_HIGH: self.support()[1]}


class GaussianSampler(Sampler):
    def __init__(self,
                 mean: float,
                 std: float) -> None:
        super().__init__()
        self.check_parameters(mean=mean, std=std)
        self._distribution = scipy.stats.norm(loc=mean, scale=std)

    @staticmethod
    def check_parameters(mean: float, std: float) -> None:
        if isinstance(mean, np.ndarray) and mean.size == 1:
            mean = mean[0]
        if isinstance(std, np.ndarray) and std.size == 1:
            std = std[0]
        assert isinstance(mean, float), 'mean must be a float.'
        assert isinstance(std, float), 'std must be a float.'
        assert std >= 0, 'std must be non-negative.'

    def save(self) -> dict:
        return {KEY_SAMPLER_OBJECT: KEY_SAMPLER_TYPE_GAUSSIAN,
                KEY_SAMPLER_GAUSSIAN_MEAN: self.expected_value(),
                KEY_SAMPLER_GAUSSIAN_STD: self.standard_deviation()}


class MixtureGaussianSampler(Sampler):  # TODO: this should become another class MixtureSampler
    def __init__(self,
                 mean: list,
                 std: list) -> None:
        super().__init__()
        self.check_parameters(mean=mean, std=std)
        self._mean = np.asarray(mean)
        self._std = np.asarray(std)
        self._distribution = []
        for i in range(len(mean)):
            self._distribution.append(scipy.stats.norm(loc=mean[i], scale=std[i]))

    @staticmethod
    def check_parameters(mean: list, std: list) -> None:
        if isinstance(mean, np.ndarray):
            mean = mean.tolist()
        if isinstance(std, np.ndarray):
            std = std.tolist()
        assert isinstance(mean, list), 'mean must be a float.'
        assert isinstance(std, list), 'std must be a float.'
        assert len(mean) == len(std), 'The length of mean and std must be consistent.'
        for s in std:
            assert s >= 0, 'Alle entries of std must be non-negative.'

    def sample(self,
               number: int,
               seed: int or None = None) -> np.ndarray:
        which = np.random.randint(low=0, high=len(self.distribution()), size=number)
        samples = np.zeros(number)
        for i, d in enumerate(self.distribution()):
            samples += np.multiply(d.rvs(size=number, random_state=seed), which == i)
        return samples

    def expected_value(self) -> float:
        return float(np.mean(np.asarray([d.mean() for d in self.distribution()])))

    def expected_values(self) -> List[float]:
        return [float(d.mean()) for d in self.distribution()]

    def standard_deviation(self) -> float:
        raise ValueError('Not implemented.')

    def standard_deviations(self) -> List[float]:
        raise [float(d.std()) for d in self.distribution()]

    def variance(self) -> float:
        raise ValueError('Not implemented.')

    def variances(self) -> List[float]:
        raise [float(d.variances()) for d in self.distribution()]

    def support(self) -> tuple:
        return min(self.distribution()), max(self.distribution())

    def save(self) -> dict:
        return {KEY_SAMPLER_OBJECT: KEY_SAMPLER_TYPE_MIXTURE_GAUSSIAN,
                KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN: self.expected_values(),
                KEY_SAMPLER_MIXTURE_GAUSSIAN_STD: self.standard_deviations()}


class UniformSamplerRecommendation(UniformSampler, SamplerRecommendation):
    def __init__(self,
                 low: float,
                 high: float) -> None:
        super().__init__(low=low, high=high)


class UniformSamplerOpinion(UniformSampler, SamplerOpinion):
    def __init__(self,
                 low: float,
                 high: float) -> None:
        super().__init__(low=low, high=high)


class GaussianSamplerRecommendation(GaussianSampler, SamplerRecommendation):
    def __init__(self,
                 mean: float,
                 std: float) -> None:
        super().__init__(mean=mean, std=std)


class GaussianSamplerOpinion(GaussianSampler, SamplerOpinion):
    def __init__(self,
                 mean: float,
                 std: float) -> None:
        super().__init__(mean=mean,
                         std=std)


class MixtureGaussianSamplerRecommendation(MixtureGaussianSampler, SamplerRecommendation):
    def __init__(self,
                 mean: list,
                 std: list) -> None:
        super().__init__(mean=mean, std=std)


class MixtureGaussianSamplerOpinion(MixtureGaussianSampler, SamplerOpinion):
    def __init__(self,
                 mean: list,
                 std: list) -> None:
        super().__init__(mean=mean, std=std)


def load_import_recommendation_sampler(parameters:dict) -> Sampler:
    if not isinstance(parameters, dict) or KEY_SAMPLER_TYPE not in parameters:
        raise ValueError('Unknown input type.')
    if KEY_SAMPLER_TYPE in parameters:
        key_objection = KEY_SAMPLER_OBJECT
    else:
        key_objection = None
    if parameters[KEY_SAMPLER_TYPE] == KEY_SAMPLER_TYPE_UNIFORM:
        if key_objection == KEY_SAMPLER_OBJECT_RECOMMENDATION:
            return UniformSamplerRecommendation(low=parameters[KEY_SAMPLER_UNIFORM_LOW],
                                                high=parameters[KEY_SAMPLER_UNIFORM_HIGH])
        elif key_objection == KEY_SAMPLER_OBJECT_RECOMMENDATION:
            return UniformSamplerOpinion(low=parameters[KEY_SAMPLER_UNIFORM_LOW],
                                         high=parameters[KEY_SAMPLER_UNIFORM_HIGH])
        else:
            return UniformSampler(low=parameters[KEY_SAMPLER_UNIFORM_LOW],
                                  high=parameters[KEY_SAMPLER_UNIFORM_HIGH])
    elif parameters[KEY_SAMPLER_TYPE] == KEY_SAMPLER_TYPE_GAUSSIAN:
        if key_objection == KEY_SAMPLER_OBJECT_RECOMMENDATION:
            return GaussianSamplerRecommendation(mean=parameters[KEY_SAMPLER_GAUSSIAN_MEAN],
                                                 std=parameters[KEY_SAMPLER_GAUSSIAN_STD])
        elif key_objection == KEY_SAMPLER_OBJECT_OPINION:
            return GaussianSamplerOpinion(mean=parameters[KEY_SAMPLER_GAUSSIAN_MEAN],
                                          std=parameters[KEY_SAMPLER_GAUSSIAN_STD])
        else:
            return GaussianSampler(mean=parameters[KEY_SAMPLER_GAUSSIAN_MEAN],
                                   std=parameters[KEY_SAMPLER_GAUSSIAN_STD])
    elif parameters[KEY_SAMPLER_TYPE] == KEY_SAMPLER_TYPE_MIXTURE_GAUSSIAN:
        if key_objection == KEY_SAMPLER_OBJECT_RECOMMENDATION:
            return GaussianSamplerRecommendation(mean=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN],
                                                 std=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_STD])
        elif key_objection == KEY_SAMPLER_OBJECT_OPINION:
            return GaussianSamplerOpinion(mean=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN],
                                          std=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_STD])
        else:
            return GaussianSampler(mean=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_MEAN],
                                   std=parameters[KEY_SAMPLER_MIXTURE_GAUSSIAN_STD])
    else:
        raise ValueError('Unknown distribution.')
