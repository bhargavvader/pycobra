# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.cobra import cobra
from pycobra.diagnostics import diagnostics
from pycobra.visualisation import visualisation

import logging

class TestVisualisation(unittest.TestCase):
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
        self.cobra_vis = visualisation(self.cobra, self.test_data, self.test_response)

    def test_indice_info(self):

        indices, mse = self.cobra_vis.indice_info(self.test_data, self.test_response, epsilon=self.cobra.epsilon)
        expected_indices, expected_mse = ('tree', 'random_forest'), 0.11358087549204622
        self.assertEqual(expected_indices, indices[0])
        self.assertAlmostEqual(expected_mse, mse[0])



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()