# Licensed under the MIT License - https://opensource.org/licenses/MIT

import numpy as np

from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation
from matplotlib.testing.decorators import image_comparison
import matplotlib
matplotlib.use('agg')

def set_up():
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
    test_data = X_test
    test_response = Y_test
    cobra_vis = Visualisation(cobra, test_data[0:4], test_response[0:4])
    return cobra_vis

@image_comparison(baseline_images=['qq'], extensions=['png'])
def test_QQ():
    cobra_vis = set_up()
    vis = cobra_vis.QQ()
    vis.plot()

@image_comparison(baseline_images=['color'], extensions=['png'])
def test_color_cobra():
    cobra_vis = set_up()
    indices, mse = cobra_vis.indice_info(epsilon=cobra_vis.aggregate.epsilon)
    colors = {('lasso', 'ridge'): "red", ('ridge', 'lasso'): "red", ('lasso', 'tree'): "green", 
    ('tree','lasso'): "green", ('random_forest','tree'): "blue", ('tree', 'random_forest'): "blue",
    ('ridge', 'lasso', 'random_forest'): "yellow", ('random_forest', 'ridge', 'lasso'): "yellow",
    ('ridge', 'random_forest', 'lasso'): "yellow", ('random_forest', 'lasso', 'ridge'): "yellow",
    ('lasso', 'ridge', 'random_forest'): "yellow", ('lasso', 'random_forest', 'ridge'): "yellow"}
    vis = cobra_vis.color_cobra(indice_info=indices, single=True, machine_colors=colors)
    vis.plot()

@image_comparison(baseline_images=['machines'], extensions=['png'])
def test_machines():
    cobra_vis = set_up()
    vis = cobra_vis.plot_machines(machines=["COBRA"], colors=["red", "green"])
    vis.plot()


