# Licensed under the MIT License - https://opensource.org/licenses/MIT

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GridSearchCV

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.cobra')


class Cobra(BaseEstimator):
    """
    COBRA: A combined regression strategy.
    Based on the paper by Biau, Fischer, Guedj, Malley [2016], this is a pythonic implementation of the original COBRA code.
    """
    def __init__(self, random_state=None, epsilon=None, X_epsilon=None, y_epsilon=None, grid_points=None):
        """
        Parameters
        ----------
        random_state: integer or a numpy.random.RandomState object.
            Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
            reproducibility of your experiments, for example.
        
        epsilon: float, optional
            Epsilon value described in the paper which determines which points are selected for the aggregate.
            Default value is determined by optimizing over a grid if test data is provided.
            If not, a mean of the possible distances is chosen.

        X_epsilon : shape = [n_samples, n_features]
            Used if no epsilon is passed to find the optimal epsilon for data passed.

        y_epsilon : array-like, shape = [n_samples]
            Used if no epsilon is passed to find the optimal epsilon for data passed.

        grid_points: int, optional
            If no epsilon value is passed, this parameter controls how many points on the grid to traverse.
        
        Attributes
        ----------
        machines: A dictionary which maps machine names to the machine objects.
                The machine object must have a predict method for it to be used during aggregation.
        machine_predictions: A dictionary which maps machine name to it's predictions over X_l
                This value is used to determine which points from y_l are used to aggregate.
        
        all_predictions: numpy array with all the predictions, to be used for epsilon manipulation.
        """
        self.machines = {}
        self.random_state = random_state
        self.epsilon = epsilon
        
        # if no epsilon value is passed, we set up COBRA to perform CV and find an optimal epsilon.
        if epsilon is None and X_epsilon is not None:
            self.X = X_epsilon
            self.y = y_epsilon
            self.split_data()
            self.load_default()
            self.load_machine_predictions()
            a, size = sorted(self.all_predictions), len(self.all_predictions)
            res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
            emin = min(res)
            emax = max(a) - min(a)
            erange = np.linspace(emin, emax, grid_points)
            tuned_parameters = [{'epsilon': erange}]
            clf = GridSearchCV(self, tuned_parameters, cv=5, scoring="neg_mean_squared_error")
            clf.fit(X_epsilon, y_epsilon)
            self.epsilon = clf.best_params_["epsilon"]
            self.machines, self.machine_predictions = {}, {}

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
        self.X = X
        self.y = y
        self.X_k = X_k
        self.X_l = X_l
        self.y_k = y_k
        self.y_l = y_l

        # set-up COBRA with default machines
        if default:
            self.split_data()
            self.load_default()
            self.load_machine_predictions()

        return self


    def pred(self, X, alpha, info=False):
        """
        Performs the COBRA aggregation scheme, used in predict method.
        
        Parameters
        ----------
        X: array-like, [n_features]

        alpha: int, optional
            alpha refers to the number of machines the prediction must be close to to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        avg: prediction

        """

        # dictionary mapping machine to points selected
        select = {}
        for machine in self.machines:
            # machine prediction
            val = self.machines[machine].predict(X)
            select[machine] = set()
            # iterating from l to n
            # replace with numpy iteration
            for count in range(0, len(self.X_l)):
                try:
                    # if value is close to prediction, select the indice
                    if math.fabs(self.machine_predictions[machine][count] - val) <= self.epsilon:
                        select[machine].add(count)
                except ValueError:
                    logger.log("Value Error")
                    continue

        points = []
        # count is the indice number. 
        for count in range(0, len(self.X_l)):
            # row check is number of machines which picked up a particular point
            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check == alpha:
                points.append(count)    
        
        # if no points are selected, return 0
        if len(points) == 0:
            if info:
                logger.info("No points were selected, prediction is 0")
                return (0, 0)
            return 0

        # aggregate
        avg = 0 
        for point in points:
            avg += self.y_l[point]
        avg = avg / len(points)

        if info:
            return avg, points
        return avg


    def predict(self, X, alpha=None, info=False):
        """
        Performs the COBRA aggregation scheme, calls pred.
        
        Parameters
        ----------
        X: array-like, [n_features]

        alpha: int, optional
            alpha refers to the number of machines the prediction must be close to to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        result: prediction

        """

        # sets alpha as the total number of machines as a default value

        X = check_array(X)

        if alpha is None:
            alpha = len(self.machines)
        if X.ndim == 1:
            return self.pred(X.reshape(1, -1), info=info, alpha=alpha)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pred(vector.reshape(1, -1), info=info, alpha=alpha)
                avg_points += len(points)
            else:
                result[index] = self.pred(vector.reshape(1, -1), info=info, alpha=alpha)              
            index += 1

        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points

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
            self.X, self.y = shuffle(self.X, self.y, random_state=self.random_state)

        if k is None and l is None:
            k = int(len(self.X) / 2)
            l = int(len(self.X))

        if k is not None and l is None:
            l = len(self.X) - k

        if l is not None and k is None:
            k = len(self.X) - l

        self.X_k = self.X[:k]
        self.X_l = self.X[k:l]
        self.y_k = self.y[:k]
        self.y_l = self.y[k:l]

        return self


    def load_default(self, machine_list=['lasso', 'tree', 'ridge', 'random_forest']):
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
        for machine in machine_list:
            if machine == 'lasso':
                self.machines['lasso'] = linear_model.LassoCV(random_state=self.random_state).fit(self.X_k, self.y_k)
            if machine == 'tree':  
                self.machines['tree'] = DecisionTreeRegressor(random_state=self.random_state).fit(self.X_k, self.y_k)
            if machine == 'ridge':
                self.machines['ridge'] = linear_model.RidgeCV().fit(self.X_k, self.y_k)
            if machine == 'random_forest':
                self.machines['random_forest'] = RandomForestRegressor(random_state=self.random_state).fit(self.X_k, self.y_k)

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
        
        self.machines[machine_name] = machine

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
        self.machine_predictions = {}
        self.all_predictions = np.array([])
        if predictions is None:
            for machine in self.machines:
                self.machine_predictions[machine] = self.machines[machine].predict(self.X_l)
                # all_predictions is used in the diagnostics class, and for initialising epsilon
                self.all_predictions = np.append(self.all_predictions, self.machine_predictions[machine])     

        if predictions is not None:
            self.machine_predictions = predictions

        return self
