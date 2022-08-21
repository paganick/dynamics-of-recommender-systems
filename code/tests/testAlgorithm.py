import unittest
import numpy as np
from modules.algorithms import UtilityMatrix
from modules.samplers import UniformSamplerRecommendation
from modules.basic import Reward


class TestAlgorithms(unittest.TestCase):
    def testUtilityMatrixExploration(self):
        for _ in range(10):
            recommendation_sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)),
                                                                  high=float(np.random.rand(1)))
            f = np.random.randint(1, 1000,1)
            p = float(np.random.rand(1))
            u = UtilityMatrix(n_agents=np.random.randint(2, 1000,1),
                              recommendation_sampler=recommendation_sampler,
                              exploration_frequency=f,
                              exploration_probability=p)
            horizon = 10000
            std_exploration = np.sqrt(p*(1.0-p))
            mean_exploration = p
            expected_count = int(horizon*1.0/f + (1.0-1.0/f)*horizon*mean_exploration)
            count = 0
            for t in range(horizon):
                explore = u.explore(time=t)
                count += explore
                if t % f == 0:  # should explore
                    self.assertTrue(explore, 'Exploration frequency not working at time ' + str(t) + '.')
            error = np.abs(count - expected_count)/((1.0-1.0/f)*horizon)
            self.assertTrue(error <= 3.0*std_exploration/np.sqrt((1.0-1.0/f)*horizon), 'Exploration probability failed (can happen with p = 0.02), here the error is ' + str(error) + '.')  # probability 99.7 %

    def testUtilityMatrix(self):
        for _ in range(10):
            # do experiment
            recommendation_sampler = UniformSamplerRecommendation(low=-float(np.random.rand(1)),
                                                                  high=float(np.random.rand(1)))
            f = np.random.randint(1, 1000,1)
            p = float(np.random.rand(1))
            n = int(np.random.randint(2, 100, 1))
            u = UtilityMatrix(n_agents=n,
                              recommendation_sampler=recommendation_sampler,
                              exploration_frequency=f,
                              exploration_probability=p)
            horizon = 100
            best_rewards = None
            best_recommendation = None
            last_recommendation = None
            for t in range(horizon):
                # generate random reward
                rewards = Reward(np.random.rand(n))
                # fix seed
                seed = int(np.random.randint(1, 100, 1))
                # implementation
                np.random.seed(seed)
                r = u.compute_recommendation(rewards, time=t)
                # different implementation
                np.random.seed(seed)
                if t == 0:
                    r_test = recommendation_sampler.sample(number=n)
                    last_recommendation = r_test.copy()
                elif u.explore(time=t):  # exploration tested above
                    r_test = recommendation_sampler.sample(number=n)
                    last_recommendation = r_test.copy()
                else:
                    if best_rewards is None:
                        best_rewards = rewards.copy()
                        best_recommendation = last_recommendation.copy()
                    else:
                        for i in range(n):
                            if rewards[i] >= best_rewards[i]:
                                best_rewards[i] = rewards[i]
                                best_recommendation[i] = last_recommendation[i]
                    r_test = best_recommendation.copy()
                    last_recommendation = r_test.copy()
                error = np.max(np.abs(r - r_test))
                self.assertAlmostEqual(error, 0, 8, 'The two recommendation should coincide, time ' + str(t) + '".')