import unittest
import numpy as np
from modules.rewardsFunctions import RewardFunctionExponential, RewardFunctionSquaredExponential
from modules.parameters import ParametersUser, ParametersPopulation, load_parameters_user, load_parameters_population
from modules.basic import Opinion


class TestInputOutput(unittest.TestCase):
    def testParameters(self):
        for _ in range(10):
            reward_1 = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            reward_2 = RewardFunctionExponential(decay_parameter=float(np.random.rand(1)))

            par_1 = ParametersUser(prejudice=Opinion(0.1),
                                   weight_prejudice=0.3,
                                   weight_recommendation=0.4,
                                   weight_current_opinion=0.3,
                                   reward=reward_1)
            par_2 = ParametersUser(prejudice=Opinion(0.0),
                                   weight_prejudice=0.2,
                                   weight_recommendation=0.1,
                                   weight_current_opinion=0.7,
                                   reward=reward_1)
            par_3 = ParametersUser(prejudice=Opinion(0.0),  # this is the same as par_2 with reward_2
                                   weight_prejudice=0.2,
                                   weight_recommendation=0.1,
                                   weight_current_opinion=0.7,
                                   reward=reward_2)
            par_4 = ParametersUser(prejudice=Opinion(0.0),  # this is the same as par_2
                                   weight_prejudice=0.2,
                                   weight_recommendation=0.1,
                                   weight_current_opinion=0.7,
                                   reward=reward_1)

            par_pop_id_1 = ParametersPopulation(parameters=par_1,
                                                repeat=1000)
            par_pop_id_2 = ParametersPopulation(parameters=par_1,
                                                repeat=100)
            par_pop_id_3 = ParametersPopulation(parameters=par_2,
                                                repeat=1000)
            par_pop_id_4 = ParametersPopulation(parameters=par_4,
                                                repeat=1000)
            par_pop_non_id_1 = ParametersPopulation(parameters=[par_1, par_2, par_3])
            par_pop_non_id_2 = ParametersPopulation(parameters=[par_1, par_1, par_3])
            par_pop_non_id_3 = ParametersPopulation(parameters=[par_1, par_4, par_3])

            self.assertNotEqual(par_1, par_2, 'The two parameters should not coincide.')
            self.assertNotEqual(par_1, par_3, 'The two parameters should not coincide.')
            self.assertEqual(par_2, par_4, 'The two parameters should  coincide.')

            self.assertNotEqual(par_pop_id_1, par_pop_id_2, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_id_1, par_pop_id_3, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_id_2, par_pop_id_4, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_id_3, par_pop_id_4, 'The two parameters should coincide.')

            self.assertNotEqual(par_pop_non_id_1, par_pop_id_1, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_non_id_1, par_pop_non_id_2, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_non_id_2, par_pop_non_id_3, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_non_id_1, par_pop_non_id_3, 'The two parameters should coincide.')

            # Input and output
            par_1_new = load_parameters_user(par_1.save())
            par_2_new = load_parameters_user(par_2.save())
            par_3_new = load_parameters_user(par_3.save())
            par_4_new = load_parameters_user(par_4.save())
            par_pop_id_1_new = load_parameters_population(par_pop_id_1.save())
            par_pop_id_2_new = load_parameters_population(par_pop_id_2.save())
            par_pop_id_3_new = load_parameters_population(par_pop_id_3.save())
            par_pop_id_4_new = load_parameters_population(par_pop_id_4.save())
            par_pop_non_id_1_new = load_parameters_population(par_pop_non_id_1.save())
            par_pop_non_id_2_new = load_parameters_population(par_pop_non_id_2.save())
            par_pop_non_id_3_new = load_parameters_population(par_pop_non_id_3.save())

            self.assertEqual(par_1, par_1_new, 'The two parameters should  coincide.')
            self.assertEqual(par_2, par_2_new, 'The two parameters should  coincide.')
            self.assertEqual(par_3, par_3_new, 'The two parameters should  coincide.')
            self.assertEqual(par_4, par_4_new, 'The two parameters should  coincide.')

            self.assertNotEqual(par_1_new, par_2_new, 'The two parameters should not coincide.')
            self.assertNotEqual(par_1_new, par_3_new, 'The two parameters should not coincide.')
            self.assertEqual(par_2_new, par_4_new, 'The two parameters should  coincide.')

            self.assertEqual(par_pop_id_1, par_pop_id_1_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_id_2, par_pop_id_2_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_id_3, par_pop_id_3_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_id_4, par_pop_id_4_new, 'The two parameters should not coincide.')

            self.assertNotEqual(par_pop_id_1_new, par_pop_id_2_new, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_id_1_new, par_pop_id_3_new, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_id_2_new, par_pop_id_4_new, 'The two parameters should coincide.')

            self.assertEqual(par_pop_non_id_1, par_pop_non_id_1_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_non_id_2, par_pop_non_id_2_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_non_id_3, par_pop_non_id_3_new, 'The two parameters should not coincide.')

            self.assertNotEqual(par_pop_non_id_1_new, par_pop_id_1_new, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_non_id_1_new, par_pop_non_id_2_new, 'The two parameters should not coincide.')
            self.assertNotEqual(par_pop_non_id_2_new, par_pop_non_id_3_new, 'The two parameters should not coincide.')
            self.assertEqual(par_pop_non_id_1_new, par_pop_non_id_3_new, 'The two parameters should coincide.')
