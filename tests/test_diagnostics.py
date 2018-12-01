# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import pytest
import numpy as np

from pycobra.cobra import Cobra
from pycobra.ewa import Ewa

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

        cobra = Cobra(random_state=0, epsilon=0.5)
        cobra.fit(X_train, Y_train)

        ewa = Ewa(random_state=0)
        ewa.fit(X_train, Y_train)

        self.test_data = X_test
        self.test_response = Y_test
        self.cobra = cobra
        self.ewa = ewa
        self.cobra_diagnostics = Diagnostics(self.cobra)
        self.cobra_diagnostics_ewa = Diagnostics(self.ewa)

    def test_alpha(self):
        alpha, mse = self.cobra_diagnostics.optimal_alpha(self.test_data, self.test_response)
        expected_alpha, expected_mse = 5, 0.06666225011910944
        self.assertEqual(expected_alpha, alpha)
        self.assertAlmostEqual(expected_mse, mse)

    @pytest.mark.slow
    def test_alpha_grid(self):
        (alpha, epsilon), mse = self.cobra_diagnostics.optimal_alpha_grid(self.test_data[0], self.test_response[0])
        expected_alpha, expected_mse = 3, 4.750490070210303e-08
        self.assertEqual(expected_alpha, alpha)
        self.assertAlmostEqual(expected_mse, mse[0])
    
    @pytest.mark.slow
    def test_machines_grid(self):
        (machines, epsilon), mse = self.cobra_diagnostics.optimal_machines_grid(self.test_data[0], self.test_response[0])
        expected_machines, expected_mse = ('ridge',), 0.00026522376609884802
        self.assertEqual(sorted(expected_machines), sorted(machines))
        self.assertAlmostEqual(expected_mse, mse[0])

    def test_machines(self):
        machines, mse = self.cobra_diagnostics.optimal_machines(self.test_data, self.test_response)
        expected_machines, expected_mse = ('ridge','tree', 'random_forest', 'svm'), 0.06511607554895288
        self.assertEqual(sorted(expected_machines), sorted(machines))
        self.assertAlmostEqual(expected_mse, mse)

    def test_epsilon(self):
        epsilon, mse = self.cobra_diagnostics.optimal_epsilon(self.test_data, self.test_response)
        expected_epsilon, expected_mse = 0.4822728142251743, 0.0650833593654587
        self.assertAlmostEqual(expected_epsilon, epsilon)
        self.assertAlmostEqual(expected_mse, mse)

    def test_split(self):
        split, mse = self.cobra_diagnostics.optimal_split(self.test_data, self.test_response)
        expected_split, expected_mse = (0.5, 0.5), 0.06666225011910944
        self.assertEqual(expected_split, split)
        self.assertAlmostEqual(expected_mse, mse)

    def test_beta(self):
        beta, mse = self.cobra_diagnostics_ewa.optimal_beta(self.test_data, self.test_response)
        expected_beta, expected_mse = 0.1, 0.07838339131485009
        self.assertAlmostEqual(expected_beta, beta)
        self.assertAlmostEqual(expected_mse, mse)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()