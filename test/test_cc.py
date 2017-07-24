# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest
import numpy as np

from pycobra.classifiercobra import ClassifierCobra
import logging
from sklearn import datasets


class TestPrediction(unittest.TestCase):
    def setUp(self):
        # setting up our random data-set
        rng = np.random.RandomState(42)
        bc = datasets.load_breast_cancer()
        self.X = bc.data[:-20]
        self.y = bc.target[:-20]
        self.test_data = bc.data[-20:]
        self.cc = ClassifierCobra(random_state=0).fit(self.X, self.y)

    def test_cc_predict(self):
        expected = 1
        result = self.cc.predict(self.test_data[0].reshape(1, -1)[0])
        self.assertEqual(expected, result)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()