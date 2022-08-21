from modules.algorithms import UtilityMatrix
from modules.agents import User
from modules.rewardsFunctions import RewardFunctionSquaredExponential
from modules.samplers import UniformSamplerRecommendation
from modules.parameters import ParametersUser
from modules.simulator import Simulator
from modules.utils import Opinion

# Parameters
reward = RewardFunctionSquaredExponential(decay_parameter=1.0)
recommendation_sampler = UniformSamplerRecommendation(low=-1.0, high=1.0)

parameters = ParametersUser(prejudice=Opinion(0.0),
                            weight_prejudice=0.2,
                            weight_recommendation=0.3,
                            weight_current_opinion=0.5,
                            reward=reward)

# Define user
user = User(initial_state=Opinion(0.0),
            parameters=parameters,
            save_history=True)

# Define algorithm
alg = UtilityMatrix(n_agents=1,
                    recommendation_sampler=recommendation_sampler,
                    exploration_probability=0.0,
                    exploration_frequency=100)

# Simulation
simulator = Simulator(agent=user,
                      algorithm=alg)

# Run
simulator.run(horizon=1000)

# Plot
simulator.agent.plot(save=False)
