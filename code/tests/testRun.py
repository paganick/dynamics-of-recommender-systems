import unittest
import os
import numpy as np
from modules.algorithms import UtilityMatrix
from modules.agents import Population, User
from modules.simulator import Simulator
from modules.rewardsFunctions import RewardFunctionExponential, RewardFunctionSquaredExponential
from modules.samplers import UniformSamplerRecommendation, GaussianSamplerRecommendation, MixtureGaussianSamplerRecommendation
from modules.samplers import UniformSamplerOpinion, GaussianSamplerOpinion, MixtureGaussianSamplerOpinion
from modules.parameters import ParametersUser, ParametersPopulation


class TestRun(unittest.TestCase):
    def setUp(self):
        self.test_directory = 'tests/plots_simulations'
        self.show_plots = False

        self.reward = RewardFunctionSquaredExponential(decay_parameter=1.0)
        self.recommendation_sampler = MixtureGaussianSamplerRecommendation(mean=[-np.random.rand(1), np.random.rand(1)],
                                                                           std=[np.random.rand(1), np.random.rand(1)])
        self.opinion_sampler_1 = UniformSamplerOpinion(low=-2*np.random.rand(1), high=0.0)
        self.opinion_sampler_2 = UniformSamplerOpinion(low=-0.0, high=2*np.random.rand(1))

        self.parameters_1 = ParametersUser(prejudice=0.0,
                                           weight_prejudice=0.0,
                                           weight_recommendation=0.4,
                                           weight_current_opinion=0.6,
                                           reward=self.reward)
        self.parameters_population_1 = ParametersPopulation(parameters=self.parameters_1,
                                                            repeat=np.random.randint(low=2, high=2e3, size=1))
        self.population_1 = Population(parameters=self.parameters_population_1,
                                       initial_state=self.opinion_sampler_1,
                                       save_history=True)
        self.parameters_2 = ParametersUser(prejudice=0.0,
                                           weight_prejudice=0.0,
                                           weight_recommendation=0.6,
                                           weight_current_opinion=0.4,
                                           reward=self.reward)
        parameters_population_2 = ParametersPopulation(parameters=self.parameters_2,
                                                       repeat=np.random.randint(low=2, high=2e3, size=1))
        self.population_2 = Population(parameters=parameters_population_2,
                                       initial_state=self.opinion_sampler_2,
                                       save_history=True)

    def tearDown(self):
        if os.path.isdir(self.test_directory):
            for file in os.listdir(self.test_directory):
                if file.endswith(('.png', '.jpeg', '.pdf', '.eps', '.h5')):
                    os.remove(os.path.join(self.test_directory, file))
            os.rmdir(self.test_directory)

    def testPopulation(self):
        population = self.population_1
        alg = UtilityMatrix(n_agents=population.n_agents(),
                            recommendation_sampler=self.recommendation_sampler,
                            exploration_probability=0.1,
                            exploration_frequency=50)
        simulator = Simulator(agent=population,
                              algorithm=alg)
        simulator.run(horizon=100, initialize=True)
        simulator.agent.plot(save=True, show=False, folder=self.test_directory, name='test_population')
        simulator.agent.save_trajectory_to_file(name='data_population', folder='results')

    def testPopulations(self):
        population = self.population_1 + self.population_2
        alg = UtilityMatrix(n_agents=population.n_agents(),
                            recommendation_sampler=self.recommendation_sampler,
                            exploration_probability=0.1,
                            exploration_frequency=50)
        simulator = Simulator(agent=population,
                              algorithm=alg)
        simulator.run(horizon=100, initialize=True)
        simulator.agent.plot(save=True, show=self.show_plots, folder=self.test_directory, name='test_populations')
        simulator.agent.save_trajectory_to_file(name='data_populations', folder=self.test_directory)

    def testUser(self):
        user = User(parameters=self.parameters_1,
                    initial_state=self.opinion_sampler_1,
                    save_history=True)
        alg = UtilityMatrix(n_agents=1,
                            recommendation_sampler=self.recommendation_sampler,
                            exploration_probability=0.1,  # probability of exploring at every time step
                            exploration_frequency=10)  # force exploration every exploration_frequency steps
        simulator = Simulator(agent=user,
                              algorithm=alg)
        simulator.run(horizon=100, initialize=True)
        simulator.agent.plot(save=True, show=False, folder=self.test_directory, name='test_user')
        simulator.agent.save_trajectory_to_file(name='data_user', folder=self.test_directory)
