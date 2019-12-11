# Licensed under the MIT License - https://opensource.org/licenses/MIT
from sklearn.metrics import mean_squared_error, accuracy_score

import itertools
import numpy as np
from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.classifiercobra import ClassifierCobra
from pycobra.kernelcobra import KernelCobra

import math
import logging

logger = logging.getLogger('pycobra.diagnostics')


class Diagnostics():
    """
    Optimization of hyperparameters, and error details.
    """
    def __init__(self, aggregate, X_test=None, y_test=None, load_MSE=False, random_state=None):
        """
        Parameters
        ----------
        aggregate: pycobra.cobra.Cobra or pycobra.cobra.Ewa object
            aggregate on which we want to run our analysis on.

        X_test : array-like, shape = [n_samples, n_features], optional.
            Testing data.

        y_test : array-like, shape = [n_samples], optional.
            Test data target values.

        load_MSE: bool, optional
            loads MSE and error bound values into diagnostics object.

        random_state: integer or a numpy.random.RandomState object.
            Set the state of the random number generator to pass on to shuffle and loading machines, to ensure
            reproducibility of your experiments, for example.
        """
        self.aggregate = aggregate
        # X_test and y_test is only to compare MSEs of machines.
        self.X_test = X_test
        self.y_test = y_test

        if load_MSE and X_test is not None and type(self.aggregate) is not ClassifierCobra:
            self.load_MSE()

        if type(self.aggregate) is ClassifierCobra and load_MSE:
            self.load_errors()

        self.random_state = random_state
        if self.random_state is None:
            self.random_state = self.aggregate.random_state


    def load_MSE(self):
        """
        Computes MSE and error bound for each Machine based on test data.

        Returns
        -------
        self : returns an instance of self.

        """
        self.machine_test_results = {}
        self.machine_MSE = {}

       
        names_dict = {Cobra: "Cobra", Ewa: "EWA"}

        for name in names_dict:
            if type(self.aggregate) is name:
                self.machine_test_results[names_dict[name]] = self.aggregate.predict(self.X_test)
                self.machine_MSE[names_dict[name]] = mean_squared_error(self.y_test, self.machine_test_results[names_dict[name]])            


        for machine in self.aggregate.estimators_:
            self.machine_test_results[machine] = self.aggregate.estimators_[machine].predict(self.X_test)
            # add MSE
            self.machine_MSE[machine] = mean_squared_error(self.y_test, self.machine_test_results[machine])

        # COBRA bound error
        power = - 2 / (len(self.aggregate.estimators_) + 2)
        self.error_bound = math.pow(len(self.aggregate.X_l_), power)

        return self

    def load_errors(self):
        """
        Computes accuracy score for each Machine based on test data.

        Returns
        -------
        self : returns an instance of self.

        """
        self.machine_test_results = {}
        self.machine_error = {}

        self.machine_test_results["ClassifierCobra"] = self.aggregate.predict(self.X_test)
        self.machine_error["ClassifierCobra"] = 1 - accuracy_score(self.y_test, self.machine_test_results["ClassifierCobra"])

        for machine in self.aggregate.estimators_:
            self.machine_test_results[machine] = self.aggregate.estimators_[machine].predict(self.X_test)
            # add MSE
            self.machine_error[machine] = 1 - accuracy_score(self.y_test, self.machine_test_results[machine])

    def optimal_alpha(self, X, y, single=False, epsilon=None, info=False):
        """
        Find the optimal alpha for testing data for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want optimal alpha values

        y: float
            Target value for query to compare.

        single: boolean, optional
            Option to calculate optimal alpha for a single query point instead.

        info: bool, optional
            Returns MSE dictionary for each alpha value

        epsilon: float, optional
            fixed epsilon value to help determine optimal alpha.

        Returns
        -------

        MSE: dictionary mapping alpha with mean squared errors
        opt: optimal alpha combination

        """
        if epsilon is None:
            epsilon = self.aggregate.epsilon

        MSE = {}
        for alpha in range(1, len(self.aggregate.estimators_) + 1):
            machine = Cobra(random_state=self.random_state, epsilon=epsilon)
            machine.fit(self.aggregate.X_, self.aggregate.y_)
            # for a single data point
            if single:
                result = machine.predict(X, alpha=alpha)
                MSE[alpha] = np.square(y - result)
            else:
                results = machine.predict(X, alpha=alpha)
                MSE[alpha] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_machines(self, X, y, single=False, epsilon=None, info=False):
        """
        Find the optimal combination of machines for testing data for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want optimal machine combinations.

        y: float
            Target value for query to compare.

        single: boolean, optional
            Option to calculate optimal machine combinations for a single query point instead.

        info: bool, optional
            Returns MSE dictionary for each machine combination value

        epsilon: float, optional
            fixed epsilon value to help determine optimal machines.

        Returns
        -------

        MSE: dictionary mapping machines with mean squared errors
        opt: optimal machines combination

        """
        if epsilon is None:
            epsilon = self.aggregate.epsilon

        n_machines = np.arange(1, len(self.aggregate.estimators_) + 1)
        MSE = {}
        for num in n_machines:
            machine_names = self.aggregate.estimators_.keys()
            use = list(itertools.combinations(machine_names, num))
            for combination in use:
                machine = Cobra(random_state=self.random_state, epsilon=epsilon)
                machine.fit(self.aggregate.X_, self.aggregate.y_, default=False)
                machine.split_data()
                machine.load_default(machine_list=combination)
                machine.load_machine_predictions()
                if single:
                    result = machine.predict(X.reshape(1, -1))
                    MSE[combination] = np.square(y - result)
                else:
                    results = machine.predict(X)
                    MSE[combination] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_epsilon(self, X, y, line_points=200, info=False):
        """
        Find the optimal epsilon value for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want for optimal epsilon.

        y: float
            Target value for query to compare.

        line_points: integer, optional
            Number of epsilon values to traverse the grid.

        info: bool, optional
            Returns MSE dictionary for each epsilon value.

        Returns
        -------

        MSE: dictionary mapping epsilon with mean squared errors
        opt: optimal epsilon value

        """

        a, size = sorted(self.aggregate.all_predictions_), len(self.aggregate.all_predictions_)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)

        MSE = {}
        for epsilon in erange:
            machine = Cobra(random_state=self.random_state, epsilon=epsilon)
            machine.fit(self.aggregate.X_, self.aggregate.y_)
            results = machine.predict(X)
            MSE[epsilon] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_split(self, X, y, split=None, epsilon=None, info=False, graph=False):

        """
        Find the optimal combination split (D_k, D_l) for fixed epsilon value for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want for optimal split.

        y: float
            Target value for query to compare.

        epsilon: float, optional.
            fixed epsilon value to help determine optimal machines.

        split: list, optional.
            D_k, D_l break-up to calculate MSE

        info: bool, optional.
            Returns MSE dictionary for each split.

        graph: bool, optional.
            Plots graph of MSE vs split

        Returns
        -------

        MSE: dictionary mapping split with mean squared errors
        opt: optimal epsilon value

        """
        if epsilon is None:
            epsilon = self.aggregate.epsilon

        if split is None:
            split = [(0.20, 0.80), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.80, 0.20)]

        MSE = {}
        for k, l in split:
            machine = Cobra(random_state=self.random_state, epsilon=epsilon)
            machine.fit(self.aggregate.X_, self.aggregate.y_, default=False)
            machine.split_data(int(k * len(self.aggregate.X_)), int((k + l) * len(self.aggregate.X_)))
            machine.load_default()
            machine.load_machine_predictions()
            results = machine.predict(X)
            MSE[(k, l)] = (mean_squared_error(y, results))

        if graph:
            import matplotlib.pyplot as plt
            ratio, mse = [], []
            for value in split:
                ratio.append(value[0])
                mse.append(MSE[value])
            plt.plot(ratio, mse)

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_alpha_grid(self, X, y, line_points=200, info=False):
        """
        Find the optimal epsilon and alpha for a single query point for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want optimal alpha and epsilon values

        y: float
            Target value for query to compare.

        line_points: integer, optional
            Number of epsilon values to traverse the grid.

        info: bool, optional
            Returns MSE dictionary for each epsilon/alpha value

        Returns
        -------

        MSE: dictionary mapping (alpha, epsilon) with mean squared errors
        opt: optimal epislon/alpha combination

        """

        # code to find maximum and minimum distance between predictions to create grid
        a, size = sorted(self.aggregate.all_predictions_), len(self.aggregate.all_predictions_)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)
        n_machines = np.arange(1, len(self.aggregate.estimators_) + 1)
        MSE = {}

        # looping over epsilon and alpha values
        for epsilon in erange:
            for num in n_machines:
                machine = Cobra(random_state=self.random_state, epsilon=epsilon)
                machine.fit(self.aggregate.X_, self.aggregate.y_)
                result = machine.predict(X.reshape(1, -1), alpha=num)
                MSE[(num, epsilon)] = np.square(y - result)

        if info:
            return MSE

        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_machines_grid(self, X, y, line_points=200, info=False):
        """
        Find the optimal epsilon and machine-combination for a single query point for the COBRA predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want optimal machines and epsilon values

        y: float
            Target value for query to compare.

        line_points: integer, optional
            Number of epsilon values to traverse the grid.

        info: bool, optional
            Returns MSE dictionary for each epsilon/machine value.

        Returns
        -------

        MSE: dictionary mapping (machine combination, epsilon) with mean squared errors
        opt: optimal epislon/machine combination

        """

        # code to find maximum and minimum distance between predictions to create grid
        a, size = sorted(self.aggregate.all_predictions_), len(self.aggregate.all_predictions_)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)
        n_machines = np.arange(1, len(self.aggregate.estimators_) + 1)
        MSE = {}

        for epsilon in erange:
            for num in n_machines:
                machine_names = self.aggregate.estimators_.keys()
                use = list(itertools.combinations(machine_names, num))
                for combination in use:
                    machine = Cobra(random_state=self.random_state, epsilon=epsilon)
                    machine.fit(self.aggregate.X_, self.aggregate.y_, default=False)
                    machine.split_data()
                    machine.load_default(machine_list=combination)
                    machine.load_machine_predictions()
                    result = machine.predict(X.reshape(1, -1))
                    MSE[(combination, epsilon)] = np.square(y - result)

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_beta(self, X, y, betas=None, info=False):
        """
        Find the optimal beta value for the Ewa predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want for optimal beta.

        y: float
            Target value for query to compare.

        betas: list, optional
            List of beta values to iterate over for optimal beta.

        info: bool, optional
            Returns MSE dictionary for each beta value.

        Returns
        -------

        MSE: dictionary mapping epsilon with mean squared errors
        opt: optimal beta value

        """

        if betas is None:
            betas = np.arange(0.1, 10)

        MSE = {}
        for beta in betas:
            machine = Ewa(random_state=self.random_state, beta=beta)
            machine.fit(self.aggregate.X_, self.aggregate.y_)
            results = machine.predict(X)
            MSE[beta] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_kernelbandwidth(self, X, y, bandwidths=None, info=False):
        """
        Find the optimal bandwidth value for the KernelCobra predictor.

        Parameteres
        -----------

        X: array-like, [n_features]
            Vector for which we want for optimal bandwidths.

        y: float
            Target value for query to compare.

        bandwidths: list, optional
            List of bandwidth values to iterate over for optimal bandwidth.

        info: bool, optional
            Returns MSE dictionary for each bandwidth value.

        Returns
        -------

        MSE: dictionary mapping epsilon with mean squared errors
        opt: optimal bandwidth value

        """

        if bandwidths is None:
            bandwidths = np.arange(0.05, 2.55, step=0.10)

        MSE = {}
        for bandwidth in bandwidths:
            machine = KernelCobra(random_state=self.random_state)
            machine.fit(self.aggregate.X_, self.aggregate.y_)
            results = machine.predict(X, bandwidth=bandwidth)
            MSE[bandwidth] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]
