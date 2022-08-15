import unittest
import numpy as np
from modules.parameters import ParametersUser, ParametersPopulation
from modules.agents import User, Population
from modules.utils import Opinion, ListOpinion, convert_list_to_recommendation
from modules.samplersRecommendation import UniformSamplerRecommendation
from modules.rewards import RewardFunctionSquaredExponential


class TestDynamics(unittest.TestCase):  # Test user dynamics (with random input and parameters)
    def testUser(self):
        for _ in range(10):
            # random parameters
            reward = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)), high=float(np.random.rand(1)))
            prejudice = 2*float(np.random.rand(1)-0.5)
            initial_state = 2*float(np.random.rand(1)-0.5)
            weight_prejudice = float(np.random.rand(1))
            weight_current_opinion = float(np.random.rand(1))
            weight_recommendation = float(np.random.rand(1))
            sum_weights = weight_prejudice + weight_current_opinion + weight_recommendation
            weight_prejudice = weight_prejudice/sum_weights
            weight_current_opinion = weight_current_opinion/sum_weights
            weight_recommendation = weight_recommendation/sum_weights

            parameters = ParametersUser(prejudice=Opinion(prejudice),
                                        weight_prejudice=weight_prejudice,
                                        weight_recommendation=weight_recommendation,
                                        weight_current_opinion=weight_current_opinion,
                                        reward=reward)
            # Define user
            user = User(initial_state=Opinion(initial_state),
                        parameters=parameters,
                        save_history=False)

            # define recommendation
            x = initial_state
            for t in range(100):
                r = convert_list_to_recommendation(sampler.sample(1))
                user.update_state(r)
                x = weight_prejudice*prejudice + weight_recommendation*float(r) + weight_current_opinion*x
                self.assertAlmostEqual(user.get_opinion(), x, 8, 'Incorrect update for a user.')

    def testIdenticalPopulation(self):
        n = 100
        for _ in range(10):
            # random parameters
            reward = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)), high=float(np.random.rand(1)))
            prejudice = 2*float(np.random.rand(1)-0.5)
            initial_state = 2*(np.random.rand(n)-0.5)
            weight_prejudice = float(np.random.rand(1))
            weight_current_opinion = float(np.random.rand(1))
            weight_recommendation = float(np.random.rand(1))
            sum_weights = weight_prejudice + weight_current_opinion + weight_recommendation
            weight_prejudice = weight_prejudice/sum_weights
            weight_current_opinion = weight_current_opinion/sum_weights
            weight_recommendation = weight_recommendation/sum_weights

            parameters = ParametersUser(prejudice=Opinion(prejudice),
                                        weight_prejudice=weight_prejudice,
                                        weight_recommendation=weight_recommendation,
                                        weight_current_opinion=weight_current_opinion,
                                        reward=reward)
            parameters_population_identical = ParametersPopulation(parameters,
                                                                   repeat=n)

            # Define user
            population_identical = Population(initial_state=ListOpinion(initial_state),
                                              parameters=parameters_population_identical,
                                              save_history=False)

            # define recommendation
            x = initial_state
            for t in range(100):
                r = sampler.sample(number=n)
                population_identical.update_state(r)
                x = weight_prejudice*prejudice + weight_recommendation*r + weight_current_opinion*x
                error = np.max(np.abs(population_identical.get_opinion_vector() - x))
                self.assertAlmostEqual(error, 0, 8, 'Incorrect update for a user.')

        def testNonIdenticalPopulation(self): #TODO: implement this
            for _ in range(10):
                pass