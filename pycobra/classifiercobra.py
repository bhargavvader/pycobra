# Licensed under the MIT License - https://opensource.org/licenses/MIT

from sklearn import neighbors, tree, svm
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.classifiercobra')


class ClassifierCobra(BaseEstimator):
    """
    Classification algorithm as described by MOJIRSHEIBANI [1999] Combining Classifiers via Discretization, Journal of the American Statistical Association.
    """
    def __init__(self, random_state=None):
        """
        Parameters
        ----------
        random_state: integer or a numpy.random.RandomState object. 
            Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
            reproducibility of your experiments, for example.
            
        Attributes
        ----------
        machines: A dictionary which maps machine names to the machine objects.
                The machine object must have a predict method for it to be used during aggregation.
        machine_predictions: A dictionary which maps machine name to it's predictions over X_l
                This value is used to determine which points from y_l are used to aggregate.
        """
        self.machines = {}
        self.random_state = random_state


    def fit(self, X, y, default=True, X_k=None, X_l=None, y_k=None, y_l=None):
        """
        Parameters
        ----------
        X: array-like, [n_samples, n_features]
            Training data which will be used to create ClassifierCobra.
        
        y: array-like [n_samples]
            

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
        """

        X, y = check_X_y(X, y)
        self.X = X
        self.y = y
        self.X_k = X_k
        self.X_l = X_l
        self.y_k = y_k
        self.y_l = y_l

        # try block to pass scikit-learn estimator check.
        try:
            # set-up COBRA with default machines
            if default:
                self.split_data()
                self.load_default()
                self.load_machine_predictions()
            # auto epsilon
        except ValueError:
            return self

        return self


    def pred(self, X, M, info=False):
        """
        Performs the COBRA aggregation scheme, used in predict method.
        
        Parameters
        ----------
        X: array-like, [n_features]

        M: int, optional
            M or alpha refers to the number of machines the prediction must be close to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        result: prediction

        """

        # dictionary mapping machine to points selected
        select = {}
        for machine in self.machines:
            # machine prediction
            label = self.machines[machine].predict(X)
            select[machine] = set()
            # iterating from l to n
            # replace with numpy iteration
            for count in range(0, len(self.X_l)):
                if self.machine_predictions[machine][count] == label:
                    select[machine].add(count)

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
                logger.info("No points were selected, prediction is null")
                return (None, 0)
            logger.info("No points were selected, prediction is null")
            return None

        # aggregate
        classes = {}
        for label in np.unique(self.y_l):
            classes[label] = 0 

        for point in points:
            classes[self.y_l[point]] += 1

        result = int(max(classes, key=classes.get))
        if info:
            return result, points
        return result


    def predict(self, X, M=None, info=False):
        """
        Performs the COBRA aggregation scheme, calls pred.
        
        Parameters
        ----------
        X: array-like, [n_features]

        M: int, optional
            M or alpha refers to the number of machines the prediction must be close to be considered during aggregation.

        info: boolean, optional
            If info is true the list of points selected in the aggregation is returned.

        Returns
        -------
        result: prediction

        """
        X = check_array(X)

        if M is None:
            M = len(self.machines)
        if X.ndim == 1:
            return self.pred(X.reshape(1, -1), info=info, M=M)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pred(vector.reshape(1, -1), info=info, M=M)
                avg_points += len(points)
            else:
                result[index] = self.pred(vector.reshape(1, -1), info=info, M=M)              
            index += 1

        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points

        return result


    def split_data(self, k=None, l=None, shuffle_data=True):
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

        Returns
        -------
        self : returns an instance of self.
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

        return self


    def load_default(self, machine_list=['sgd', 'tree', 'knn', 'svm']):
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
            if machine == 'svm':
                self.machines['svm'] = svm.SVC().fit(self.X_k, self.y_k)
            if machine == 'knn':  
                self.machines['knn'] = neighbors.KNeighborsClassifier().fit(self.X_k, self.y_k)
            if machine == 'sgd':
                self.machines['sgd'] = SGDClassifier(loss="hinge", penalty="l2").fit(self.X_k, self.y_k)
            if machine == 'tree':
                self.machines['tree'] = tree.DecisionTreeClassifier().fit(self.X_k, self.y_k)

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
        Stores the trained machines' predicitons on D_l in a dictionary, to be used for predictions.
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

        return self
