from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.cobra')

# when we have more util functions, we can add this there
# def get_random_state(seed):
#      """ Turn seed into a np.random.RandomState instance.

#          Method originally from maciejkula/glove-python, and written by @joshloyal
#      """
#      if seed is None or seed is np.random:
#          return np.random.mtrand._rand
#      if isinstance(seed, (numbers.Integral, np.integer)):
#          return np.random.RandomState(seed)
#      if isinstance(seed, np.random.RandomState):
#         return seed
#      raise ValueError('%r cannot be used to seed a np.random.RandomState instance' % seed)


class cobra():
    """
    COBRA - Nonlinear Aggregation of Predictors.
    
    Based on the paper by Biau, Guedj et al [2016], this is a pythonic implementation of the original COBRA code.
    """
    def __init__(self, X, y, epsilon=None, default=True, X_k=None, X_l=None, y_k=None, y_l=None, test_data=None, test_response=None, random_state=None, line_points=80):
        """
        Parameters
        ----------
        X : shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] 
            Target values

        epsilon: float, optional
            Epsilon value described in the paper which determines which points are selected for the aggregate.
            Default value is determined by running a grid if test data is provided. 
            If not, a mean of the possible distances is chosen.
        
        default: bool, optional
            If set as true then sets up COBRA with default machines and splitting.

        X_k : shape = [n_samples, n_features]
            Training data which is used to train the machines loaded into COBRA. 
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_k : array-like, shape = [n_samples]
            Target values used to train the machines loaded into COBRA.

        X_l : shape = [n_samples, n_features]
            Training data which is used during the aggregation of COBRA.
            Can be loaded directly into COBRA; if not, the split_data method is used as default.

        y_l : array-like, shape = [n_samples] 
            Target values which are actually used in the aggregation of COBRA.

        test_data: shape = [n_samples, n_features]
            Testing data used to determine optimal data-dependant epsilon value if it is not passed.

        test_response : array-like, shape = [n_samples] 
            Target values used to determine optimal data-dependant epsilon value if it is not passed.

        random_state: integer or a numpy.random.RandomState object. 
            Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
            reproducibility of your experiments, for example.

        line_points: integer, optional
            Number of epsilon values to traverse the grid.
            
        Attributes
        ----------
        
        machines: A dictionary which maps machine names to the machine objects.
                The machine object must have a predict method for it to be used during aggregation.

        machine_predictions: A dictionary which maps machine name to it's predictions over X_l
                This value is used to determine which points from y_l are used to aggregate.
        
        all_predictions: numpy array with all the predictions, to be used for epsilon manipulation.

        """
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.X_k = X_k
        self.X_l = X_l
        self.y_k = y_k
        self.y_l = y_l

        self.machines = {}

        self.random_state = random_state

        # set-up COBRA with default machines
        if default:
            self.split_data()
            self.load_default()
            self.load_machine_predictions()
        # auto epsilon without test data
        if self.epsilon is None and test_data is None:
            a, size = sorted(self.all_predictions), len(self.all_predictions)
            res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
            emin = min(res)
            emax = max(a) - min(a)
            self.epsilon = (emin + emax) / 2

        # auto epsilon with test data
        if self.epsilon is None and test_data is not None:
            from pycobra.diagnostics import diagnostics
            cobra_diagnostics = diagnostics(cobra=self, X_test=test_data, y_test=test_response, load_MSE=True)
            self.epsilon = cobra_diagnostics.optimal_epsilon(test_data, test_response, line_points=line_points)[0]


    def split_data(self, k=None, l=None, shuffle_data=False):
        """
        Split the data into different parts for training machines, and for aggregation.

        Parameters
        ----------
        k : int, optional
            k determines D_k, which is the data used to the train the machines.

        l: int, optional
            l determines D_l, which is the data used in the aggregation.

        shuffle: bool, optional
            Boolean value to decide to shuffle the data before splitting.

        random_state: numpy random_state object or int
            Random seed if shuffling.
        """

        if shuffle_data:
            self.X, self.y = shuffle(self.X, self.y, random_state=self.random_state)

        if k is None and l is None:
            k = int(len(self.X) / 2)
            l = int(len(self.X))

        self.X_k = self.X[:k]
        self.X_l = self.X[k:l]
        self.y_k = self.y[:k]
        self.y_l = self.y[k:l]


    def load_default(self, machine_list=['lasso', 'tree', 'ridge', 'random_forest']):
        """
        Loads 4 different scikit-learn regressors by default. 

        Parameters
        ----------
        machine_list: optional, list of strings
            List of default machine names to be loaded. 

        """
        for machine in machine_list:
            if machine == 'lasso':
                self.machines['lasso'] = linear_model.LassoLars().fit(self.X_k, self.y_k)
            if machine == 'tree':  
                self.machines['tree'] = DecisionTreeRegressor(random_state=self.random_state).fit(self.X_k, self.y_k)
            if machine == 'ridge':
                self.machines['ridge'] = linear_model.Ridge(random_state=self.random_state).fit(self.X_k, self.y_k)
            if machine == 'random_forest':
                self.machines['random_forest'] = RandomForestRegressor(random_state=self.random_state).fit(self.X_k, self.y_k)


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
        Stores the trained machines' predicitons on D_l in a dictionary, to be used for predictions.
        Should be run after all the machines to be used for aggregation is loaded.

        Parameters
        ----------        

        predictions: dictionary, optional
            A pre-existing machine:predictions dictionary can also be loaded.

        """
        self.machine_predictions= {}
        self.all_predictions = np.array([])
        if predictions is None:
            for machine in self.machines:
                self.machine_predictions[machine] = self.machines[machine].predict(self.X_l)
                # all_predictions is used in the diagnostics class, and for initialising epsilon
                self.all_predictions = np.append(self.all_predictions, self.machine_predictions[machine])     

        if predictions is not None:
            self.machine_predictions = predictions


    def predict(self, X, M=None, info=False):
        """
        Performs the COBRA aggregation scheme for a single input vector X.
        
        Parameters
        ----------
        X: array-like, [n_features]

        M: int, optional
            M or alpha refers to the number of machines the prediction must be close to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        avg: prediction

        """

        # sets M as the total number of machines as a default value
        if M is None:
            M = len(self.machines)

        if self.X_l is not None:
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
                if row_check == M:
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


    def predict_array(self, X_array, info=False, M=None):
        """
        Performs the COBRA aggregation scheme for an array X
        
        Parameters
        ----------
        X_array: array-like, [n_features]

        M: int, optional
            M or alpha refers to the number of machines the prediction must be close to be considered during aggregation.

        info: boolean, optional
            If info is true the average number of points selected for a single vector is returned.

        Returns
        -------
        result: array of prediction
        """
        result = np.zeros(len(X_array))
        count = 0
        avg_points = 0
        for X in X_array:
            if info:
                result[count], points = self.predict(X.reshape(1, -1), info=info, M=M)
                avg_points += len(points)
            else:
                result[count] = self.predict(X.reshape(1, -1), info=info, M=M)              
            count += 1

        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points

        return result

