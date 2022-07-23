from rewards import RewardFunctionExponential, RewardFunctionSquaredExponential
from parameters import ParametersUser, ParametersPopulation, load_parameters_user, load_parameters_population
from utils import Opinion

# Parameters
reward_1 = RewardFunctionSquaredExponential(decay_parameter=1.0)
reward_2 = RewardFunctionExponential(decay_parameter=2.0)

parameters_1 = ParametersUser(prejudice=Opinion(0.1),
                              weight_prejudice=0.3,
                              weight_recommendation=0.4,
                              weight_current_opinion=0.3,
                              reward=reward_1)
parameters_2 = ParametersUser(prejudice=Opinion(0.0),
                              weight_prejudice=0.2,
                              weight_recommendation=0.1,
                              weight_current_opinion=0.7,
                              reward=reward_2)
# check
parameters_population_identical = ParametersPopulation(parameters=parameters_1,
                                                       repeat=1000)
parameters_population_non_identical = ParametersPopulation(parameters=[parameters_1, parameters_2])

print('Parameters 1 and 1 coincide:                  ' + str(parameters_1 == parameters_1))
print('Parameters 1 and 2 coincide:                  ' + str(parameters_1 == parameters_2))
print('Parameters of identical populations coincide: ' + str(parameters_population_identical == parameters_population_identical))
print('Parameters of identical populations coincide: ' + str(parameters_population_non_identical == parameters_population_non_identical))
print('Parameters of populations coincide:           ' + str(parameters_population_identical == parameters_population_non_identical))
print('')

# save parameters
out_1 = parameters_1.save()
out_identical = parameters_population_identical.save()
out_non_identical = parameters_population_non_identical.save()

# load parameters
parameters_1_new = load_parameters_user(out_1)
parameters_population_identical_new = load_parameters_population(out_identical)
parameters_population_non_identical_new = load_parameters_population(out_non_identical)

# check if they coincide
print('User:                     ' + str(parameters_1 == parameters_1_new))
print('Population identical:     ' + str(parameters_population_identical == parameters_population_identical_new))
print('Population non-identical: ' + str(parameters_population_non_identical == parameters_population_non_identical))
