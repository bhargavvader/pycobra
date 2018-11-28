# Licensed under the MIT License - https://opensource.org/licenses/MIT

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import math
import numpy as np
import random
import logging

logger = logging.getLogger('pycobra.ewa')


class Ewa(BaseEstimator):
    """
    Exponential Weighted Average aggregation method.
    Implementation based on work by:
    M. Mojirsheibani (1999), Combining Classifiers via Discretization,
    Journal of the American Statistical Association.

    Parameters
    ----------
    random_state: integer or a numpy.random.RandomState object.
        Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
        reproducibility of your experiments, for example.

    beta: float, optional
        Parameter to be passed when creating machine weights for EWA.

    Attributes
    ----------
    machines_ : dictionary
        dictionary mapping name of machine to the object.

    machine_predictions_: A dictionary which maps machine name to it's predictions over X_l
            This value is used to determine which points from y_l are used to aggregate.
    """
    def __init__(self, random_state=None, beta=None):
        self.random_state = random_state
        self.beta = beta

    def fit(self, X, y, default=True, X_k=None, X_l=None, y_k=None, y_l=None):
        """
        Parameters
        ----------
        X: array-like, [n_features]
            Training data which will be used to create the EWA aggregate.

        y: array-like, shape = [n_samples]
            Target values used to train the machines used in the EWA aggregate.

        default: bool, optional
            If set as true then sets up EWA with default machines and splitting.

        X_k : shape = [n_samples, n_features], optional
            Training data which is used to train the machines loaded into Ewa.
            Can be loaded directly into EWA; if not, the split_data method is used as default.

        y_k : array-like, shape = [n_samples], optional
            Target values used to train the machines loaded into EWA.

        X_l : shape = [n_samples, n_features], optional
            Training data which is used during the aggregation of EWA.
            Can be loaded directly into EWA; if not, the split_data method is used as default.

        y_l : array-like, shape = [n_samples], optional
            Target values which are actually used in the aggregation of EWA.

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----

        We store the data used to train the machines because this information is used to make the prediction.

        """

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l
        self.estimators_ = {}

        # set-up Ewa with default machines
        if default:
            self.split_data()
            self.load_default()
            self.load_machine_weights(self.beta)

        return self


    def set_beta(self, X_beta=None, y_beta=None, betas=None):
        """
        Parameters
        ----------

        betas: list, optional
            List of betas to find optimal beta for weights

        X_beta : shape = [n_samples, n_features]
            Used if no beta is passed to find the optimal beta for data passed.

        y_beta : array-like, shape = [n_samples]
            Used if no beta is passed to find the optimal beta for data passed.
        """

        if self.beta is None and X_beta is not None:
            if betas is None:
                betas = [0.001, 0.01, 0.1,  1.0, 10.0, 100.0]
            tuned_parameters = [{'beta': betas}]
            clf = GridSearchCV(self, tuned_parameters, cv=5, scoring="neg_mean_squared_error")
            clf.fit(X_beta, y_beta)
            self.beta = clf.best_params_["beta"]


    def split_data(self, k=None, l=None, shuffle_data=False):
        """
        Split the data into different parts for training machines and for aggregation.

        Parameters
        ----------
        k : int, optional
            k is the number of points used to train the machines.
            Those are the first k points of the data provided.

        l: int, optional
            l is the number of points used to form the EWA aggregate.

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

        """
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


    def load_machine_weights(self, beta=None):
        """
        Loads the EWA weights for each machine based on the training data.
        Should be run after all the machines to be used for aggregation is loaded.

        Parameters
        ----------
        beta : float
            Inverse temperature parameter to form the weights.

        Returns
        -------
        self : returns an instance of self.
        """

        if self.beta is None and beta is None:
            beta = 1.0
        if self.beta is not None and beta is None:
            beta = self.beta

        self.machine_MSE_ = {}
        self.machine_weight_ = {}
        for machine in self.estimators_:
            self.machine_MSE_[machine] = mean_squared_error(self.y_l_, self.estimators_[machine].predict(self.X_l_))
            self.machine_weight_[machine] = np.exp(beta * self.machine_MSE_[machine])
            if self.machine_weight_[machine] == np.inf:
                logger.info("MSE too high, setting equal weights to all machines")
                for machine in self.estimators_:
                    self.machine_weight_[machine] = 1 / len(self.estimators_)
                return self

        normalise = sum(self.machine_weight_.values(), 0.0)
        self.machine_weight_ = {k: v / normalise for k, v in self.machine_weight_.items()}

        return self


    def predict(self, X):
        """
        Parameters
        ----------
        X: array-like, [n_features]

        Returns
        -------
        result: returns prediction
        """
        X = check_array(X)

        result = 0.0
        for machine in self.estimators_:
            result += self.machine_weight_[machine] * self.estimators_[machine].predict(X)

        return result


    def plot_machine_weights(self, figsize=8):
        """
        Plot each machine weights

        Parameteres
        -----------
        figsize: float, optional
            Size of plot.
        """

        plt.bar(range(len(self.machine_weight_)), self.machine_weight_.values(), align='center')
        plt.xticks(range(len(self.machine_weight_)), self.machine_weight_.keys())

        plt.show()
