# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.cobra import Cobra
from pycobra.ewa import Ewa

import logging

class TestPrediction(unittest.TestCase):
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
        self.cobra = cobra
        self.ewa = ewa

    def test_cobra_predict(self):
        expected = 2.7310842344617035
        result = self.cobra.predict(self.test_data[0].reshape(1, -1))
        self.assertAlmostEqual(expected, result)

    def test_ewa_predict(self):
        expected = 2.7716310515173239
        result = self.ewa.predict(self.test_data[0].reshape(1, -1))
        self.assertAlmostEqual(expected, result[0])        


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()