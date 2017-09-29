# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation

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
        # for testing
        Y_train = Y[:D1 + D2]
        Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]

        cobra = Cobra(random_state=0, epsilon=0.5)
        cobra.fit(X_train, Y_train)
        self.test_data = X_test
        self.test_response = Y_test
        self.cobra = cobra
        self.cobra_vis = Visualisation(self.cobra, self.test_data[0:4], self.test_response[0:4])

    def test_indice_info(self):

        indices, mse = self.cobra_vis.indice_info(self.test_data[0:4], self.test_response[0:4], epsilon=self.cobra.epsilon)
        expected_indices, expected_mse = ('ridge','lasso'), 0.3516475171334160
        self.assertEqual(sorted(expected_indices), sorted(indices[0]))
        self.assertAlmostEqual(expected_mse, mse[0][0])

        # we now run the visualisations using indices - does not run on travis
        # vor = self.cobra_vis.voronoi(indice_info=indices)
        # self.cobra_vis.color_cobra(indice_info=indices)

    def test_visualisations(self):
        # run all visualisation methods - does not run on travis
        # self.cobra_vis.boxplot()
        # self.cobra_vis.QQ()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()