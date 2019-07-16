# Licensed under the MIT License - https://opensource.org/licenses/MIT

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GridSearchCV

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.kernelcobra')


class KernelCobra(BaseEstimator):
    """
    Regression algorithm as introduced by
    Kernel-COBRA: A combined regression-classification strategy using Kernels.
    Based on the paper by Guedj, Srinivasa Desikan [2018].

    Parameters
    ----------
    random_state: integer or a numpy.random.RandomState object.
        Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
        reproducibility of your experiments, for example.


    Attributes
    ----------
    estimators_: A dictionary which maps machine names to the machine objects.
            The machine object must have a predict method for it to be used during aggregation.

    machine_predictions_: A dictionary which maps machine name to it's predictions over X_l
            This value is used to determine which points from y_l are used to aggregate.

    all_predictions_: numpy array with all the predictions, to be used for bandwidth manipulation.

    """

    def __init__(self, random_state=None):
        self.random_state = random_state


    def fit(self, X, y, default=True, X_k=None, X_l=None, y_k=None, y_l=None):
        """
        Parameters
        ----------
        X: array-like, [n_samples, n_features]
            Training data which will be used to create the COBRA aggregate.

        y: array-like, shape = [n_samples]
            Target values used to train the machines used in the aggregation.

        default: bool, optional
            If set as true then sets up COBRA with default machines and splitting.

        X_k : shape = [n_samples, n_features]
            Training data which is used to train the machines used in the aggregation.
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_k : array-like, shape = [n_samples]
            Target values used to train the machines used in the aggregation.

        X_l : shape = [n_samples, n_features]
            Training data which is used to form the aggregate.
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_l : array-like, shape = [n_samples]
            Target values which are actually used to form the aggregate.
        """

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l
        self.estimators_ = {}
        # set-up COBRA with default machines
        if default:
            self.split_data()
            self.load_default()
            self.load_machine_predictions()

        return self


    def pred(self, X, kernel=None, metric=None, bandwidth=1, **kwargs):
        """
        Performs the Kernel-COBRA aggregation scheme, used in predict method.

        Parameters
        ----------
        X: array-like, [n_features]

        kernel: function, optional
            kernel refers to the kernel method which we wish to use to perform the aggregation.

        metric: function, optional
            metric refers to the metric method which we wish to use to perform the aggregation.

        bandwidth: float, optional
            Bandwidth for the deafult kernel value (gaussian), and is set to 1.

        kwargs requires you to pass arguments with "kernel_params" and "metric_params", if the custom kernel or metric
        has more paramteres.        

        Returns
        -------
        avg: prediction

        """

        a = np.zeros(len(self.X_l_))
        for machine in self.estimators_:
            val = self.estimators_[machine].predict(X)
            for index, value in np.ndenumerate(self.machine_predictions_[machine]):
                if metric is not None:
                    try:
                        a[index] += metric(value, val, kwargs["metric_params"])
                    except KeyError:
                        a[index] += metric(value, val)
                else:
                    a[index] += math.fabs(value - val)

        # normalise the array
        if kernel is not None:
            try:
                a = np.divide(kernel(a, kwargs["kernel_params"]), np.sum(kernel(a, kwargs["kernel_params"])))
            except KeyError:
                a = np.divide(kernel(a), np.sum(kernel(a)))
        else:
            a = np.divide(np.exp(- bandwidth * a), np.sum(np.exp(- bandwidth * a)))

        return np.sum(np.multiply(self.y_l_, a))


    def predict(self, X, kernel=None, metric=None, bandwidth=1, **kwargs):
        """
        Performs the Kernel-COBRA aggregation scheme, calls pred.

        Parameters
        ----------
        X: array-like, [n_features]

        kernel: function, optional
            kernel refers to the kernel method which we wish to use to perform the aggregation.

        metric: function, optional
            metric refers to the metric method which we wish to use to perform the aggregation.

        bandwidth: float, optional
            Bandwidth for the deafult kernel value (gaussian), and is set to 1.

        kwargs requires you to pass arguments with "kernel_params" and "metric_params", if the custom kernel or metric
        has more paramteres.        

        Returns
        -------
        avg: prediction

        """

        X = check_array(X)

        if X.ndim == 1:
            return self.pred(X.reshape(1, -1))

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            result[index] = self.pred(vector.reshape(1, -1), kernel=kernel, metric=metric, bandwidth=bandwidth, **kwargs)
            index += 1

        return result


    def split_data(self, k=None, l=None, shuffle_data=False):
        """
        Split the data into different parts for training machines and for aggregation.

        Parameters
        ----------
        k : int, optional
            k is the number of points used to train the machines.
            Those are the first k points of the data provided.

        l: int, optional
            l is the number of points used to form the COBRA aggregate.

        shuffle: bool, optional
            Boolean value to decide to shuffle the data before splitting.

        Returns
        -------
        self : returns an instance of self.
        """

        if shuffle_data:
            self.X_, self.y_ = shuffle(self.X_, self.y_, random_state=self.random_state)

        if k is None and l is None:
            k = int(len(self.X_) / 2)
            l = int(len(self.X_))

        if k is not None and l is None:
            l = len(self.X_) - k

        if l is not None and k is None:
            k = len(self.X_) - l

        self.X_k_ = self.X_[:k]
        self.X_l_ = self.X_[k:l]
        self.y_k_ = self.y_[:k]
        self.y_l_ = self.y_[k:l]

        return self


    def load_default(self, machine_list=['lasso', 'tree', 'ridge', 'random_forest', 'svm']):
        """
        Loads 4 different scikit-learn regressors by default.

        Parameters
        ----------
        machine_list: optional, list of strings
            List of default machine names to be loaded.
        Returns
        -------
        self : returns an instance of self.
        """
        self.estimators_ = {}
        for machine in machine_list:
            if machine == 'lasso':
                self.estimators_['lasso'] = linear_model.LassoCV(random_state=self.random_state).fit(self.X_k_, self.y_k_)
            if machine == 'tree':
                self.estimators_['tree'] = DecisionTreeRegressor(random_state=self.random_state).fit(self.X_k_, self.y_k_)
            if machine == 'ridge':
                self.estimators_['ridge'] = linear_model.RidgeCV().fit(self.X_k_, self.y_k_)
            if machine == 'random_forest':
                self.estimators_['random_forest'] = RandomForestRegressor(random_state=self.random_state).fit(self.X_k_, self.y_k_)
            if machine == 'svm':
                self.estimators_['svm'] = SVR().fit(self.X_k_, self.y_k_)

        return self


    def load_machine(self, machine_name, machine):
        """
        Adds a machine to be used during the aggregation strategy.
        The machine object must have been trained using X_k and y_k, and must have a 'predict()' method.
        After the machine is loaded, for it to be used during aggregation, load_machine_predictions must be run.

        Parameters
        ----------
        machine_name : string
            Name of the machine you are loading

        machine: machine/regressor object
            The regressor machine object which is mapped to the machine_name

        Returns
        -------
        self : returns an instance of self.
        """

        self.estimators_[machine_name] = machine

        return self


    def load_machine_predictions(self, predictions=None):
        """
        Stores the trained machines' predicitons on training data in a dictionary, to be used for predictions.
        Should be run after all the machines to be used for aggregation is loaded.

        Parameters
        ----------
        predictions: dictionary, optional
            A pre-existing machine:predictions dictionary can also be loaded.

        Returns
        -------
        self : returns an instance of self.
        """
        self.machine_predictions_ = {}
        self.all_predictions_ = np.array([])
        if predictions is None:
            for machine in self.estimators_:
                self.machine_predictions_[machine] = self.estimators_[machine].predict(self.X_l_)
                # all_predictions_ is used in the diagnostics class, and for initialising epsilon
                self.all_predictions_ = np.append(self.all_predictions_, self.machine_predictions_[machine])

        if predictions is not None:
            self.machine_predictions_ = predictions

        return self
