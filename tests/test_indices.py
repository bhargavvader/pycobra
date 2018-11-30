# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation

import logging
import matplotlib
matplotlib.use('agg')


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
        self.indices, self.mse = self.cobra_vis.indice_info(self.test_data[0:4], self.test_response[0:4], epsilon=self.cobra.epsilon)

        ewa = Ewa(random_state=0)
        ewa.fit(X_train, Y_train)
        self.ewa = ewa
        self.ewa_vis = Visualisation(self.ewa, self.test_data[0:4], self.test_response[0:4])

    def test_indice_info(self):
        expected_indices, expected_mse = ('ridge', 'lasso', 'svm'), 0.3516475171334160
        self.assertEqual(sorted(expected_indices), sorted(self.indices[0]))
        self.assertAlmostEqual(expected_mse, self.mse[0][0])

    def test_voronoi(self):

        vor = self.cobra_vis.voronoi(indice_info=self.indices)
        min_bound, max_bound = -0.19956180892237763, 0.9046027692022134
        self.assertAlmostEqual(min_bound, vor.min_bound[0])
        self.assertAlmostEqual(max_bound, vor.max_bound[0])

        vor_ = self.cobra_vis.voronoi(indice_info=self.indices, single=True)
        min_bound, max_bound = -0.19956180892237763, 0.9046027692022134
        self.assertAlmostEqual(min_bound, vor_.min_bound[0])
        self.assertAlmostEqual(max_bound, vor_.max_bound[0])

    def test_boxplot(self):

        expected_data_len =  100
        data = self.cobra_vis.boxplot(info=True)
        self.assertEqual(len(data[0]), expected_data_len)

        data = self.ewa_vis.boxplot(info=True)
        self.assertEqual(len(data[0]), expected_data_len)

    def test_QQ(self):
        self.cobra_vis.QQ()

    def test_color_cobra(self):
        self.cobra_vis.color_cobra(indice_info=self.indices, single=True)

    def test_machines(self):
        self.cobra_vis.plot_machines()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()