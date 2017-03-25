from sklearn.metrics import mean_squared_error
import itertools
import numpy as np
from pycobra.cobra import cobra
import math

import logging

logger = logging.getLogger('pycobra.diagnostics')

class diagnostics():
    """
    Optimization of parameters, and error details. 
    """
    def __init__ (self, cobra, X_test, y_test, load_MSE=True, random_state=None):
        """
        Parameters
        ----------
        cobra: pycobra.cobra object
            cobra object on which we want to run our analysis on.

        X_test : array-like, shape = [n_samples, n_features], optional.
            Testing data.

        y_test : array-like, shape = [n_samples], optional.
            Test data target values.
        
        load_MSE: bool, optional
            loads MSE and error bound values into diagnostics object.
        """
        self.cobra = cobra
        self.X_test = X_test
        self.y_test = y_test

        if load_MSE:
            self.load_MSE()

        self.random_state = random_state
        if self.random_state is None:
            self.random_state = self.cobra.random_state


    def load_MSE(self):
        """
        Computes MSE and error bound for each Machine based on test data.
        """
        self.machine_test_results = {}
        self.machine_MSE = {}

        self.machine_test_results["COBRA"] = self.cobra.predict_array(self.X_test)
        self.machine_MSE["COBRA"] = mean_squared_error(self.y_test, self.machine_test_results["COBRA"])
        
        for machine in self.cobra.machines:
            self.machine_test_results[machine] = self.cobra.machines[machine].predict(self.X_test)
            # add MSE
            self.machine_MSE[machine] = mean_squared_error(self.y_test, self.machine_test_results[machine])

        # COBRA bound error
        power = - 2 / (len(self.cobra.machines) + 2)
        self.error_bound = math.pow(len(self.cobra.X_l), power) 


    def optimal_alpha(self, X, y, single=False, epsilon=None, info=False):
        """
        Find the optimal alpha for testing data.

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want optimal alpha values
        
        y_test: float
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
            epsilon = self.cobra.epsilon

        MSE = {}
        for alpha in range(1, len(self.cobra.machines) + 1):
                machine = cobra(self.X_test, self.y_test, epsilon=epsilon, random_state=self.random_state)
                # for a single data point
                if single:
                    result = machine.predict(X.reshape(1, -1), M=alpha) 
                    MSE[alpha] = np.square(y - result)
                else:
                    results = machine.predict_array(X, M=alpha)
                    MSE[alpha] = (mean_squared_error(y, results))

        if info:
            return MSE      
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_machines(self, X, y, single=False, epsilon=None, info=False):
        """
        Find the optimal combination of machines for testing data.

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want optimal machine combinations.
        
        y_test: float
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
            epsilon = self.cobra.epsilon

        n_machines = np.arange(1, len(self.cobra.machines) + 1)
        MSE = {}
        for num in n_machines:
            machine_names = self.cobra.machines.keys()
            use = list(itertools.combinations(machine_names, num))
            for combination in use:
                machine = cobra(self.X_test, self.y_test, epsilon = epsilon, random_state=self.random_state, default=False)
                machine.split_data()
                machine.load_default(machine_list=combination)
                machine.load_machine_predictions() 
                if single:
                    result = machine.predict(X.reshape(1, -1)) 
                    MSE[combination] = np.square(y - result)
                else:
                    results = machine.predict_array(X)
                    MSE[combination] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_epsilon(self, X, y, line_points=200, info=False):
        """
        Find the optimal combination epsilon value for X_test

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want for optimal epsilon.
        
        y_test: float
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

        a, size = sorted(self.cobra.all_predictions), len(self.cobra.all_predictions)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)

        MSE = {}
        for epsilon in erange:
            machine = cobra(self.X_test, self.y_test, epsilon=epsilon, random_state=self.random_state)
            results = machine.predict_array(X)
            MSE[epsilon] = (mean_squared_error(y, results))

        if info:
            return MSE
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_split(self, X, y, split=None, epsilon=None, info=False, graph=False):
        
        """
        Find the optimal combination split (D_k, D_l) for fixed epsilon value.

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want for optimal split.
        
        y_test: float
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
            epsilon = self.cobra.epsilon

        if split is None:
            split = [(0.20, 0.80), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.80, 0.20)]

        MSE = {}
        for k, l in split:
            machine = cobra(self.X_test, self.y_test, epsilon=epsilon, random_state=self.random_state, default=False)
            machine.split_data(int(k * len(self.X_test)), int((k + l) * len(self.X_test)))
            machine.load_default()
            machine.load_machine_predictions() 
            results = machine.predict_array(X)
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
        Find the optimal epsilon and alpha for a single query point.

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want optimal alpha and epsilon values
        
        y_test: float
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
        a, size = sorted(self.cobra.all_predictions), len(self.cobra.all_predictions)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)
        n_machines = np.arange(1, len(self.cobra.machines) + 1)
        MSE = {}

        # looping over epsilon and alpha values
        for epsilon in erange:
            for num in n_machines:
                machine = cobra(self.X_test, self.y_test, epsilon=epsilon, random_state=self.random_state)
                result = machine.predict(X.reshape(1, -1), M=num)
                MSE[(num, epsilon)] = np.square(y - result)
    
        if info:
            return MSE

        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]


    def optimal_machines_grid(self, X, y, line_points=200, info=False):
        """
        Find the optimal epsilon and machine-combination for a single query point.

        Parameteres
        -----------

        X_test: array-like, [n_features]
            Vector for which we want optimal machines and epsilon values
        
        y_test: float
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
        a, size = sorted(self.cobra.all_predictions), len(self.cobra.all_predictions)
        res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
        emin = min(res)
        emax = max(a) - min(a)
        erange = np.linspace(emin, emax, line_points)
        n_machines = np.arange(1, len(self.cobra.machines) + 1)
        MSE = {}

        for epsilon in erange:
            for num in n_machines:
                machine_names = self.cobra.machines.keys()
                use = list(itertools.combinations(machine_names, num))
                for combination in use:
                    machine = cobra(self.X_test, self.y_test, epsilon = epsilon, random_state=self.random_state, default=False)
                    machine.split_data()
                    machine.load_default(machine_list=combination)
                    machine.load_machine_predictions() 
                    result = machine.predict(X.reshape(1, -1))
                    MSE[(combination, epsilon)] = np.square(y - result)

        if info:
            return MSE      
        opt = min(MSE, key=MSE.get)
        return opt, MSE[opt]

