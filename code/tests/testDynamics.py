import unittest
import numpy as np
from modules.parameters import ParametersUser, ParametersPopulation
from modules.agents import User, Population
from modules.basic import Opinion
from modules.samplers import UniformSamplerRecommendation
from modules.rewardsFunctions import RewardFunctionSquaredExponential
from modules.utils import KEY_OPINION, KEY_RECOMMENDATION, KEY_REWARD


class TestDynamics(unittest.TestCase):  # Test user dynamics (with random input and parameters)
    def testUser(self):
        for _ in range(10):
            # random parameters
            reward = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)), high=float(np.random.rand(1)))
            prejudice = 2*(np.random.rand(1)-0.5)
            initial_state = 2*(np.random.rand(1)-0.5)
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
            user = User(parameters=parameters,
                        initial_state=Opinion(initial_state),
                        save_history=False)

            # define recommendation
            horizon = 100
            x = initial_state
            x_hist = np.array([x])
            reward_hist = np.array([np.nan])
            recommendation_hist = np.array([np.nan])
            for t in range(horizon):
                if np.random.rand() <= 0.05 and t <= horizon-2:
                    x_new = 2*(np.random.rand(1)-0.5)
                    x = x_new
                    x_hist = x_new.copy().reshape((1, -1))
                    reward_hist, recommendation_hist = np.array([np.nan]), np.array([np.nan])
                    user.initialize(initial_state=x_new)
                else:
                    r = sampler.sample(number=1)
                    reward = user.update_state(r)
                    x = weight_prejudice*prejudice + weight_recommendation*r + weight_current_opinion*x
                    x_hist = np.concatenate((x_hist, np.asarray(x).reshape((1, -1))))
                    if np.any(np.isnan(reward_hist)):
                        reward_hist = reward.reshape((1, -1))
                    else:
                        reward_hist = np.concatenate((reward_hist, reward.reshape((1, -1))))
                    if np.any(np.isnan(recommendation_hist)):
                        recommendation_hist = r.reshape((1, -1))
                    else:
                        recommendation_hist = np.concatenate((recommendation_hist, r.reshape((1, -1))))
                r = sampler.sample(1)
                user.update_state(r)
                x = weight_prejudice*prejudice + weight_recommendation*float(r) + weight_current_opinion*x
                self.assertAlmostEqual(float(user.opinion()), float(x), 8, 'Incorrect update for a user.')

    def testIdenticalPopulation(self):
        n = 100
        for _ in range(10):
            # random parameters
            reward = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)), high=float(np.random.rand(1)))
            prejudice = 2*(np.random.rand(1)-0.5)
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
            population_identical = Population(parameters=parameters_population_identical,
                                              initial_state=Opinion(initial_state),
                                              save_history=True)

            # run
            horizon = 100
            x = initial_state
            x_hist = np.array([x])
            reward_hist = np.array([np.nan])
            recommendation_hist = np.array([np.nan])
            for t in range(horizon):
                if np.random.rand() <= 0.05 and t <= horizon-2:
                    x_new = 2*(np.random.rand(n)-0.5)
                    x = x_new
                    x_hist = x_new.copy().reshape((1, -1))
                    reward_hist, recommendation_hist = np.array([np.nan]), np.array([np.nan])
                    population_identical.initialize(initial_state=x_new)
                else:
                    r = sampler.sample(number=n)
                    reward = population_identical.update_state(r)
                    x = weight_prejudice*prejudice + weight_recommendation*r + weight_current_opinion*x
                    x_hist = np.concatenate((x_hist, np.asarray(x).reshape((1, -1))))
                    if np.any(np.isnan(reward_hist)):
                        reward_hist = reward.reshape((1, -1))
                    else:
                        reward_hist = np.concatenate((reward_hist, reward.reshape((1, -1))))
                    if np.any(np.isnan(recommendation_hist)):
                        recommendation_hist = r.reshape((1, -1))
                    else:
                        recommendation_hist = np.concatenate((recommendation_hist, r.reshape((1, -1))))
                error = np.max(np.abs(population_identical.opinions() - x))
                self.assertAlmostEqual(error, 0, 8, 'Incorrect update for a user at time ' + str(t) + '.')
            error_1 = np.max(np.max(np.abs(population_identical.trajectory[KEY_OPINION] - x_hist)))
            error_2 = np.max(np.max(np.abs(population_identical.trajectory[KEY_REWARD] - reward_hist)))
            error_3 = np.max(np.max(np.abs(population_identical.trajectory[KEY_RECOMMENDATION] - recommendation_hist)))
            self.assertAlmostEqual(error_1, 0, 8, 'Incorrect trajectories (opinion).')
            self.assertAlmostEqual(error_2, 0, 8, 'Incorrect trajectories (reward).')
            self.assertAlmostEqual(error_3, 0, 8, 'Incorrect trajectories (recommendation).')

        def testInitialization(self):  # TODO: implement this
            for _ in range(10):
                pass

        def testNonIdenticalPopulation(self): # TODO: implement this
            for _ in range(10):
                pass