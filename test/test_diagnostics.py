import unittest
import numpy as np

from pycobra.cobra import cobra
from pycobra.diagnostics import diagnostics
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
        X_eps = X[D1 + D2:D1 + D2 + D3]
        # for testing
        Y_train = Y[:D1 + D2]
        Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]
        Y_eps = Y[D1 + D2:D1 + D2 + D3]

        COBRA = cobra(X_train, Y_train, epsilon = 0.5, random_state=0)

        COBRA.split_data(D1, D1 + D2)
        COBRA.load_default()
        COBRA.load_machine_predictions()
        self.test_data = X_test
        self.test_response = Y_test
        self.eps_data = X_eps
        self.eps_response = Y_eps
        self.cobra = COBRA
        self.cobra_diagnostics = diagnostics(self.cobra, self.test_data, self.test_response, load_MSE=True)


    def test_alpha(self):
        alpha, mse = self.cobra_diagnostics.optimal_alpha(self.test_data, self.test_response)
        expected_alpha, expected_mse = 3, 0.12008320911254405
        self.assertEqual(expected_alpha, alpha)
        self.assertAlmostEqual(expected_mse, mse)

    def test_alpha_grid(self):
        (alpha, epsilon), mse = self.cobra_diagnostics.optimal_alpha_grid(self.test_data[0], self.test_response[0])
        expected_alpha, expected_mse, expected_epsilon = 4, 0.11358087549204622, 0.40807699665207059
        self.assertEqual(expected_alpha, alpha)
        self.assertAlmostEqual(expected_mse, mse)
        self.assertAlmostEqual(expected_epsilon, epsilon)

    def test_machines(self):
        machines, mse = self.cobra_diagnostics.optimal_machines(self.test_data, self.test_response)
        expected_machines, expected_mse = ('tree', 'random_forest'), 0.068391666517022165
        self.assertEqual(expected_machines, machines)
        self.assertAlmostEqual(expected_mse, mse)

    def test_machines_grid(self):
        (machines, epsilon), mse = self.cobra_diagnostics.optimal_machines_grid(self.test_data[0], self.test_response[0])
        expected_machines, expected_mse, expected_epsilon = ('random_forest', 'lasso'), 0.10868778993795772, 0
        self.assertEqual(expected_machines, machines)
        self.assertAlmostEqual(expected_mse, mse)

    def test_epsilon(self):
        epsilon, mse = self.cobra_diagnostics.optimal_epsilon(self.test_data, self.test_response)
        expected_epsilon, expected_mse = 0.38952804225879467, 0.05745740344638442
        self.assertAlmostEqual(expected_epsilon, epsilon)
        self.assertAlmostEqual(expected_mse, mse)

    def test_split(self):
        split, mse = self.cobra_diagnostics.optimal_split(self.test_data, self.test_response)
        expected_split, expected_mse = (0.6, 0.4), 0.056748503269048782
        self.assertEqual(expected_split, split)
        self.assertAlmostEqual(expected_mse, mse)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()