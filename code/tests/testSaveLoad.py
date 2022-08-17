import unittest
import numpy as np
import os
from modules.saveUtils import save_dict_to_file, load_dict_from_data


class TestSaveLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.out = []
        self.out.append({'a': 1.0})
        self.out.append({'a': np.random.rand(100)})
        self.out.append({'c': 'test'})
        self.out.append({'a': np.random.rand(100),
                         'b': 1.0,
                         'c': 'test'})
        self.test_directory = 'tests/results'

    def tearDown(self):
        if os.path.isdir(self.test_directory):
            for i in range(len(self.out)):
                os.remove(os.path.join(self.test_directory, 'test_' + str(i+1)) + '.h5')
            os.rmdir(self.test_directory)

    def testSaveLoad(self):
        for i, o in enumerate(self.out):
            save_dict_to_file(o, name='test_' + str(i+1), folder=self.test_directory)
        for i in range(len(self.out)):
            loaded = load_dict_from_data(name='test_' + str(i+1), folder=self.test_directory)
            for key, value in loaded.items():
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.all(value == self.out[i][key]), 'The two dict should coincide.')
                else:
                    self.assertEqual(value, self.out[i][key], 'The two dict should coincide.')

    def testWrongInput(self):
        self.assertRaises(ValueError, save_dict_to_file, 1.0, name='test', folder=self.test_directory)
        self.assertRaises(ValueError, save_dict_to_file, np.random.rand(10), name='test', folder=self.test_directory)
        self.assertRaises(ValueError, save_dict_to_file, self.out[-1], name=1.0, folder=self.test_directory)
        self.assertRaises(ValueError, save_dict_to_file, self.out[-1], name='test', folder=1.0)


