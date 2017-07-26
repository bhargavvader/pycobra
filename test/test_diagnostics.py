# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
import logging

class TestOptimal(unittest.TestCase):
    def setUp(self):
        # setting up our random data-set
        rng = np.random.RandomState(42)

        # D1 = train machines; D2 = create COBRA; D3 = calibrate epsilon, alpha; D4 = testing
        n_features = 20
        D1, D2, D3, D4 = 200, 200, 200, 200
        D = D1 + D2 + D3 + D4
        X = rng.uniform(-1, 1, D * n_features).reshape(D, n_features)
        Y = np.power(X[:,1], 2) + np.power(X[:,3], 3) + np.exp(X[:,10]) 

        # training data-set
        X_train = X[:D1 + D2]
        X_test = X[D1 + D2 + D3:D1 + D2 + D3 + D4]
        # for testing
        Y_train = Y[:D1 + D2]
        Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]

        COBRA = Cobra(random_state=0, epsilon=0.5)
        COBRA.fit(X_train, Y_train)
        self.test_data = X_test
        self.test_response = Y_test
        self.cobra = COBRA
        self.cobra_diagnostics = Diagnostics(self.cobra)


    def test_alpha(self):
        alpha, mse = self.cobra_diagnostics.optimal_alpha(self.test_data, self.test_response)
        expected_alpha, expected_mse = 4, 0.068051089937708101
        self.assertEqual(expected_alpha, alpha)
        self.assertAlmostEqual(expected_mse, mse)

    # the grid tests are skipped because of long build times - to test these methods un-comment and run the tests again.
    # def test_alpha_grid(self):
    #     (alpha, epsilon), mse = self.cobra_diagnostics.optimal_alpha_grid(self.test_data[0], self.test_response[0])
    #     expected_alpha, expected_mse = 1, 0.0133166
    #     self.assertEqual(expected_alpha, alpha)
    #     self.assertAlmostEqual(expected_mse, mse[0])

    # def test_machines_grid(self):
    #     (machines, epsilon), mse = self.cobra_diagnostics.optimal_machines_grid(self.test_data[0], self.test_response[0])
    #     expected_machines, expected_mse = ('ridge',), 0.00026522376609884802
    #     self.assertEqual(sorted(expected_machines), sorted(machines))
    #     self.assertAlmostEqual(expected_mse, mse[0])

    def test_machines(self):
        machines, mse = self.cobra_diagnostics.optimal_machines(self.test_data, self.test_response)
        expected_machines, expected_mse = ('ridge','tree', 'random_forest'), 0.066681946568334649
        self.assertEqual(sorted(expected_machines), sorted(machines))
        self.assertAlmostEqual(expected_mse, mse)

    def test_epsilon(self):
        epsilon, mse = self.cobra_diagnostics.optimal_epsilon(self.test_data, self.test_response)
        expected_epsilon, expected_mse = 0.35243013347224278, 0.062335364376335425
        self.assertAlmostEqual(expected_epsilon, epsilon)
        self.assertAlmostEqual(expected_mse, mse)

    def test_split(self):
        split, mse = self.cobra_diagnostics.optimal_split(self.test_data, self.test_response)
        expected_split, expected_mse = (0.5, 0.5), 0.068051089937708101
        self.assertEqual(expected_split, split)
        self.assertAlmostEqual(expected_mse, mse)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()