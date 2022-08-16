import unittest
import numpy as np
from modules.parameters import ParametersUser
from modules.rewards import RewardFunctionSquaredExponential
from modules.utils import Opinion
from modules.agents import User


class TestUser(unittest.TestCase):
    def testEqualUsers(self):
        for _ in range(10):
            reward = RewardFunctionSquaredExponential(decay_parameter=float(np.random.rand(1)))
            par_1 = ParametersUser(prejudice=Opinion(0.1),
                                   weight_prejudice=0.3,
                                   weight_recommendation=0.4,
                                   weight_current_opinion=0.3,
                                   reward=reward)
            par_2 = ParametersUser(prejudice=Opinion(0.0),
                                   weight_prejudice=0.2,
                                   weight_recommendation=0.1,
                                   weight_current_opinion=0.7,
                                   reward=reward)
            u_1 = User(parameters=par_1,
                       initial_state=Opinion(float(np.random.rand())),
                       save_history=True)
            u_2 = User(parameters=par_2,
                       initial_state=Opinion(float(np.random.rand())),
                       save_history=True)
            u_3 = User(parameters=par_1,
                       initial_state=Opinion(float(np.random.rand())),
                       save_history=True)

            self.assertEqual(u_1, u_3, 'The two users should coincide.')
            self.assertNotEqual(u_1, u_2, 'The two users should not coincide.')
            self.assertEqual(u_1, u_1, 'The two users should coincide.')