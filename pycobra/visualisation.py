# Licensed under the MIT License - https://opensource.org/licenses/MIT

import math
import itertools

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils import shuffle

from pycobra.diagnostics import Diagnostics
from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.classifiercobra import ClassifierCobra

from collections import OrderedDict

import logging


logger = logging.getLogger('pycobra.visualisation')

def create_labels(indice_info):
    """
    Helper method to create labels for plotting.

    Parameters
    ----------

    indice_info: list of strings
        List of machine names

    Return
    ------

    label: string
        Serves as a label during plotting.
    """

    label = ""
    for machine in indice_info:
        label = machine + " + " + label
    return label[:-3] 


def gen_machine_colors(only_colors=False, num_colors=None, indice_info=None, rgb=False, plot_machines=None, colors=None):
    """
    Helper method to create a machine combinations to color dictionary, or a list of colors.
    
    Parameters
    ----------

    indice_info: dictionary, optional
        Dictionary which is a result of running pycobra.visualisation.indice_info. Maps indices to combinations of machines.
    
    only_colors: bool, optional
        Option to return only a list of colors

    num_colors: int, optional
        Number of colors to be returned if using only_colors

    rgb : bool, optional
        Creates dictionary based on machine used and r, g, b, a scheme.

    plot_machines: list of strings, optional
        List of machines to use in rgb coloring.

    colors: list of strings, optional
        List of colors to be used for pairing with machine_combinations

    Return
    ------

    machine_colors: dictionary
        Dictionary mapping machine combinations and color.
    """

    # note: for default colors to be assigned, the latest version of matplotlib is needed.
    # the code below is taken from the colors example: 
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    # if we need only as many colors as individual machine
    if only_colors:
        # make the sorted names list equal to in size as the machine combinations
        # we do this by first picking up equally spaced out colors, then randonly picking out the right amount
        sorted_names = sorted_names[0::int(len(sorted_names) / num_colors)]
        return random.sample(sorted_names, num_colors)

    machine_combinations = list(set(indice_info.values()))
    machine_colors = {}

    if rgb:
        for indice in indice_info:
            r, g, b, a = 0, 0, 0, 0.4
            if plot_machines[0] in indice_info[indice]:
                r = 1
            if plot_machines[1] in indice_info[indice]:
                g = 1
            if plot_machines[2] in indice_info[indice]:
                b = 1
            if plot_machines[3] in indice_info[indice]:
                a = 1
            if (r, g, b) == (1, 1, 1):
                r, g, b = 0, 0, 0
            machine_colors[indice_info[indice]] = (r, g, b, a)
        return machine_colors

    # if it isn't rgb, let's pair each unique machine with a color provided
    if colors is not None and len(machine_combinations) == len(colors):
        for machine, color in zip(machine_combinations, colors):
            machine_colors[machine] = color
        return machine_colors

    # if it's none of the above options, we create colors similar to the only colors option.
    sorted_names = sorted_names[0::int(len(sorted_names) / len(machine_combinations))]
    colors = random.sample(sorted_names, len(machine_combinations))

    for machine, color in zip(machine_combinations, colors):
        machine_colors[machine] = color
    return machine_colors


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Code originally written by pv: https://gist.github.com/pv/8036995.
    Helper method for voronoi.
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


class Visualisation():
    """
    Plots and visualisations of COBRA aggregates.
    If X_test and y_test is loaded, you can run the plotting functions with no parameters.
    """
    def __init__(self, aggregate, X_test, y_test, plot_size=8, random_state=None):
        """
        Parameters
        ----------
        aggregate: pycobra.cobra.Cobra or pycobra.cobra.Ewa object
            aggregate on which we want to run our analysis on.

        X_test : array-like, shape = [n_samples, n_features].
            Testing data.

        y_test : array-like, shape = [n_samples].
            Test data target values.
        
        plot_size: int, optional
            Size of matplotlib plots.

        """
        self.aggregate = aggregate
        self.X_test = X_test
        self.y_test = y_test
        self.plot_size = plot_size

        # load results so plotting doesn't need parameters
        self.machine_test_results = {}
        if type(aggregate) is Cobra:
            self.machine_test_results["COBRA"] = self.aggregate.predict(self.X_test)

        if type(aggregate) is Ewa:
            self.machine_test_results["EWA"] = self.aggregate.predict(self.X_test)
        
        if type(aggregate) is ClassifierCobra:        
            self.machine_test_results["ClassifierCobra"] = self.aggregate.predict(self.X_test)

        for machine in self.aggregate.machines:
            self.machine_test_results[machine] = self.aggregate.machines[machine].predict(self.X_test)
        
        self.random_state = random_state
        if self.random_state is None:
            self.random_state = self.aggregate.random_state


    def plot_machines(self, machines=None, colors=None, plot_indices=False):
        """
        Plot the results of the machines versus the actual answers (testing space).
        
        Parameters
        ----------
        machines: list, optional
            List of machines to plot.
        colors: list, optional
            Colors of machines.
        plot_indices: boolean, optional.
            Plots truth values against indices.

        """

        if machines is None:
            machines = self.machine_test_results.keys()

        plt.figure(figsize=(self.plot_size, self.plot_size))
        
        if plot_indices:
            linspace = np.linspace(0, len(self.y_test), len(self.y_test))     

        if colors is None:
            colors = gen_machine_colors(only_colors=True, num_colors=len(machines) + 1)

        if plot_indices:
            plt.scatter(linspace, self.y_test, color=colors[0], label="Truth") 
        else:
            plt.scatter(self.X_test, self.y_test, color=colors[0], label="Truth") 

        for machine, color in zip(machines, colors[1:]):
            if plot_indices:
                plt.scatter(linspace, self.machine_test_results[machine], color=color, label=machine)
            else:
                plt.scatter(self.X_test, self.machine_test_results[machine], color=color, label=machine)

        if plot_indices:
            plt.xlabel("Point Indice")

        plt.legend()
        plt.show()


    def QQ(self, machine="COBRA"):
        """
        Plots the machine results vs the actual results in the form of a QQ-plot.
        
        Parameters
        ----------
        machine: string, optional
            Name of machine to perform QQ-plot.
        """
        plt.figure(figsize=(self.plot_size, self.plot_size))
        axes = plt.gca()
        pred = self.machine_test_results[machine]

        # this is to make the plot look neater
        min_limits = math.fabs(min(min(pred), min(self.y_test))) 
        max_limits = max(max(pred), max(self.y_test))
        axes.set_xlim([min(min(pred), min(self.y_test)) - min_limits, max(max(pred), max(self.y_test)) + max_limits])
        axes.set_ylim([min(min(pred), min(self.y_test)) - min_limits, max(max(pred), max(self.y_test)) + max_limits])

        # scatter the machine responses versus the actual y_test
        plt.scatter(self.y_test, pred, label=machine)
        axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3")

        # labels
        plt.xlabel('RESPONSES')
        plt.ylabel('PREDICTED')

        plt.legend()
        plt.show()


    def boxplot(self, reps=100):
        """
        Plots boxplots of machines.

        Parameters
        ----------
        reps: int, optional
            Number of times to repeat experiments for boxplot.

        """
        if type(self.aggregate) is Cobra:
            
            MSE = {k: [] for k, v in self.aggregate.machines.items()}
            MSE["COBRA"] = []
            for i in range(0, reps):
                cobra = Cobra(random_state=self.random_state, epsilon=self.aggregate.epsilon)
                X, y = shuffle(self.aggregate.X, self.aggregate.y, random_state=self.aggregate.random_state)
                cobra.fit(X, y, default=False)
                cobra.split_data(shuffle_data=True)

                for machine in self.aggregate.machines:
                    self.aggregate.machines[machine].fit(cobra.X_k, cobra.y_k)
                    cobra.load_machine(machine, self.aggregate.machines[machine])
                
                cobra.load_machine_predictions()
                X_test, y_test = shuffle(self.X_test, self.y_test, random_state=self.aggregate.random_state)
                
                for machine in cobra.machines:                
                   MSE[machine].append(mean_squared_error(y_test, cobra.machines[machine].predict(X_test)))
                MSE["COBRA"].append(mean_squared_error(y_test, cobra.predict(X_test)))

            data, labels = [], []
            for machine in MSE:
                data.append(MSE[machine])
                labels.append(machine)
        
        if type(self.aggregate) is Ewa:
            
            MSE = {k: [] for k, v in self.aggregate.machines.items()}
            MSE["EWA"] = []
            for i in range(0, reps):
                ewa = Ewa(random_state=self.random_state, beta=self.aggregate.beta)
                X, y = shuffle(self.aggregate.X, self.aggregate.y, random_state=self.aggregate.random_state)
                ewa.fit(X, y, default=False)
                ewa.split_data(shuffle_data=True)
                
                for machine in self.aggregate.machines:
                    self.aggregate.machines[machine].fit(ewa.X_k, ewa.y_k)
                    ewa.load_machine(machine, self.aggregate.machines[machine])

                ewa.load_machine_weights(self.aggregate.beta)
                X_test, y_test = shuffle(self.X_test, self.y_test, random_state=self.aggregate.random_state)
                for machine in ewa.machines:                
                   MSE[machine].append(mean_squared_error(y_test, ewa.machines[machine].predict(X_test)))
                MSE["EWA"].append(mean_squared_error(y_test, ewa.predict(X_test)))

            data, labels = [], []
            for machine in MSE:
                data.append(MSE[machine])
                labels.append(machine)

        if type(self.aggregate) is ClassifierCobra:
            
            errors = {k: [] for k, v in self.aggregate.machines.items()}
            errors["ClassifierCobra"] = []
            for i in range(0, reps):
                cc = ClassifierCobra(random_state=self.random_state)
                X, y = shuffle(self.aggregate.X, self.aggregate.y, random_state=self.aggregate.random_state)
                cc.fit(X, y, default=False)
                cc.split_data(shuffle_data=True)
                
                for machine in self.aggregate.machines:
                    self.aggregate.machines[machine].fit(cc.X_k, cc.y_k)
                    cc.load_machine(machine, self.aggregate.machines[machine])

                cc.load_machine_predictions()
                X_test, y_test = shuffle(self.X_test, self.y_test, random_state=self.aggregate.random_state)
                for machine in cc.machines:                
                   errors[machine].append(1 - accuracy_score(y_test, cc.machines[machine].predict(X_test)))
                errors["ClassifierCobra"].append(1 - accuracy_score(y_test, cc.predict(X_test)))

            data, labels = [], []
            for machine in errors:
                data.append(errors[machine])
                labels.append(machine)


        plt.figure(figsize=(self.plot_size, self.plot_size))
        plt.boxplot(data, labels=labels)
        plt.show()

    def indice_info(self, X_test=None, y_test=None, epsilon=None, line_points=200):
        """
        Method to return information about each indices (query) optimal machines for testing data.

        Parameters
        ----------
        epsilon: float, optional
            Epsilon value to use for diagnostics
        
        line_points: int, optional
            if epsilon is not passed, optimal epsilon is found per point.

        Returns
        -------

        indice_info: dicitonary mapping indice to optimal machines.

        MSE: dictionary mapping indice to mean squared error for optimal machines for that point.

        """

        if X_test is None:
            X_test = self.X_test  
        if y_test is None:
            y_test = self.y_test

        indice = 0
        indice_info = {}
        MSE = {}
        cobra_diagnostics = Diagnostics(aggregate=self.aggregate, random_state=self.random_state)
        if epsilon is None:
            for data_point, response in zip(X_test, y_test):
                info = cobra_diagnostics.optimal_machines_grid(data_point, response, line_points=line_points)
                indice_info[indice], MSE[indice] = info[0][0], info[1]
                indice += 1
        else:
            for data_point, response in zip(X_test, y_test):
                info = cobra_diagnostics.optimal_machines(data_point, response, single=True, epsilon=epsilon)
                indice_info[indice], MSE[indice] = info[0], info[1]
                indice += 1

        return indice_info, MSE


    def color_cobra(self, X_test=None, y_test=None, line_points=200, epsilon=None, indice_info=None, plot_machines=["ridge", "lasso", "random_forest", "tree"], single=False, machine_colors=None):
        """
        Plot the input space and color query points based on the optimal machine used for that point.

        Parameters
        ----------
        epsilon: float, optional
            Epsilon value to use for diagnostics. Used to find indice_info if it isn't passed.
        
        line_points: int, optional
            if epsilon is not passed, optimal epsilon is found per point. Used to find indice_info if it isn't passed.

        indice_info: dicitonary, optional 
            dictionary mapping indice to optimal machines.

        plot_machines: list, optional
            list of machines to be plotted.

        single: bool, optional
            plots a single plot with each machine combination.

        machine_colors: dictionary, optional
            Depending on the kind of coloring, a dictionary mapping machines to colors.

        """

        if indice_info is None:
            indice_info = self.indice_info(line_points, epsilon)
  
        if X_test is None:
            X_test = self.X_test  
        if y_test is None:
            y_test = self.y_test      
        # we want to plot only two columns
        data_1 = X_test[:,0]
        data_2 = X_test[:,1]

        if single:
            if machine_colors is None:
                machine_colors = gen_machine_colors(indice_info=indice_info)
            plt.ion()
            fig, ax = plt.subplots()
            plot = ax.scatter([], [])
            for indice in indice_info:
                
                ax.set_title("All Machines")
                ax.scatter(data_1[indice], data_2[indice], color=machine_colors[indice_info[indice]], label=create_labels(indice_info[indice]))

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        if not single:
            if machine_colors is None:
                machine_colors = {}
                colors = gen_machine_colors(only_colors=True, num_colors=len(plot_machines))
                for machine, color in zip(plot_machines, colors):
                    machine_colors[machine] = color

            for machine in plot_machines:
                plt.ion()
                fig, ax = plt.subplots()
                plot = ax.scatter([], [])
                # set boundaries based on the data
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_title(machine)
                for indice in indice_info:
                    if machine in indice_info[indice]:
                        ax.scatter(data_1[indice], data_2[indice], color=machine_colors[machine])


    def voronoi(self, X_test=None, y_test=None, line_points=200, epsilon=None, indice_info=None, MSE=None, plot_machines=["ridge", "lasso", "random_forest", "tree"], machine_colors=None, gradient=False, single=False):
        """
        Plot the input space and color query points as a Voronoi Tesselation based on the optimal machine used for that point.

        Parameters
        ----------
        epsilon: float, optional
            Epsilon value to use for diagnostics. Used to find indice_info if it isn't passed.
        
        line_points: int, optional
            if epsilon is not passed, optimal epsilon is found per point. Used to find indice_info if it isn't passed.

        indice_info: dicitonary, optional 
            dictionary mapping indice to optimal machines.
        
        MSE: dictionary, optional
            dictionary mapping indice to mean-squared error for optimal machines

        plot_machines: list, optional
            list of machines to be plotted.

        single: bool, optional
            plots a single plot with each machine combination.
 
        gradient: bool, optional
            instead of aggregating optimal machines, plots a colored plot for each machine,
            shaded according to the mean-squared error of that "region"

        machine_colors: dictionary, optional
            Depending on the kind of coloring, a dictionary mapping machines to colors. 
        """

        if X_test is None:
            X_test = self.X_test  
        if y_test is None:
            y_test = self.y_test

        if indice_info is None:
            indice_info, MSE = self.indice_info(line_points, epsilon)
            
        # passing input space to set up voronoi regions.
        points = np.hstack((np.reshape(X_test[:,0], (len(X_test[:,0]), 1)), np.reshape(X_test[:,1], (len(X_test[:,1]), 1))))
        vor = Voronoi(points)
        # use helper Voronoi
        regions, vertices = voronoi_finite_polygons_2d(vor)
        
        # # colorize
        if not single:
            for machine in plot_machines:
                fig, ax = plt.subplots()
                plot = ax.scatter([], [])
                ax.set_title(machine)
                indice = 0
                for region in regions:
                    ax.plot(X_test[:,0][indice], X_test[:,1][indice], 'ko')
                    polygon = vertices[region]
                    if gradient is True and MSE is not None:
                    # we find closest index from range to give gradient value
                        mse_range = np.linspace(min(MSE.values()), max(MSE.values()), 10)
                        num = min(mse_range, key=lambda x:abs(x - MSE[indice]))
                        index = np.where(mse_range == num)
                        alpha = index[0][0] / 10.0 + 0.2
                        if alpha > 1.0:
                            alpha = 1.0
                    # we fill the polygon with the appropriate gradient
                        if machine in indice_info[indice]:
                            ax.fill(*zip(*polygon), alpha=alpha, color='r')
                    else:
                    # if it isn't gradient based we just color red or blue depending on whether that point uses the machine in question
                        if machine in indice_info[indice]:
                            ax.fill(*zip(*polygon), alpha=0.4, color='r', label="")
                        else:
                            ax.fill(*zip(*polygon), alpha=0.4, color='b', label="")
                    indice += 1

                ax.axis('equal')
                plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
                plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

            return vor

        if single:

            if machine_colors is None:
                machine_colors = gen_machine_colors(indice_info=indice_info)

            fig, ax = plt.subplots()
            plot = ax.scatter([], [])
            ax.set_title("All Machines")
            indice = 0
            for region in regions:

                ax.plot(X_test[:,0][indice], X_test[:,1][indice], 'ko')
                polygon = vertices[region]
                ax.fill(*zip(*polygon), alpha=0.2, color=machine_colors[indice_info[indice]], label=create_labels(indice_info[indice]))
                indice += 1

            ax.axis('equal')
            plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
            plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)   
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            return vor
