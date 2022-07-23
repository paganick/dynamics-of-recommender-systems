import numpy as np
from algorithms import UtilityMatrix
from agents import User
from rewards import RewardFunctionExponential, RewardFunctionSquaredExponential
from parameters import ParametersUser
from simulator import Simulator
from utils import Opinion

# Parameters
reward = RewardFunctionSquaredExponential(decay_parameter=1.0)

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
                    exploration_probability=0.0,
                    exploration_frequency=100)

# Simulation
simulator = Simulator(agent=user,
                      algorithm=alg)

# Run
simulator.run(horizon=1000)

# Plot
simulator.agent.plot(save=False)
