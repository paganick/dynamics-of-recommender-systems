import numpy as np
from algorithms import UtilityMatrix
from agents import Population
from rewards import RewardFunctionExponential, RewardFunctionSquaredExponential
from parameters import ParametersUser, ParametersPopulation
from utils import Opinion, ListOpinion
from simulator import Simulator

# Parameters
reward_1 = RewardFunctionSquaredExponential(decay_parameter=1.0)
reward_2 = RewardFunctionExponential(decay_parameter=2.0)

parameters_1 = ParametersUser(prejudice=Opinion(0.0),
                              weight_prejudice=0.0,
                              weight_recommendation=0.1,
                              weight_current_opinion=0.9,
                              reward=reward_1)
parameters_2 = ParametersUser(prejudice=Opinion(0.0),
                              weight_prejudice=0.0,
                              weight_recommendation=0.1,
                              weight_current_opinion=0.9,
                              reward=reward_2)

parameters_population_identical = ParametersPopulation(parameters=parameters_1,
                                                       repeat=1000)
parameters_population_non_identical = ParametersPopulation(parameters=[parameters_1, parameters_2])

# Define population
population_identical = Population(initial_state=ListOpinion(np.random.uniform(low=-1.0, high=1.0, size=1000)),
                                  parameters=parameters_population_identical,
                                  save_history=True)
population_non_identical = Population(initial_state=ListOpinion(0.5*np.ones(2)),
                                      parameters=parameters_population_non_identical,
                                      save_history=True)

# Define algorithm
alg_identical = UtilityMatrix(n_agents=population_identical.n_agents(),
                              exploration_probability=0.0,
                              exploration_frequency=10)
alg_non_identical = UtilityMatrix(n_agents=population_non_identical.n_agents(),
                                  exploration_probability=0.0,
                                  exploration_frequency=10)

# Simulators
simulator_identical = Simulator(agent=population_identical,
                                algorithm=alg_identical)
simulator_non_identical = Simulator(agent=population_non_identical,
                                    algorithm=alg_non_identical)

# Run
simulator_identical.run(horizon=1000)
# simulator_non_identical.run(horizon=1000)

# Plot
simulator_identical.agent.plot(save=False, name='sim_identical')
# simulator_non_identical.agent.plot(save=False, name='sim_non_identical')


