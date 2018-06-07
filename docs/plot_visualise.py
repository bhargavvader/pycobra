"""
COBRA Visualisations
--------------------

This notebook will cover the visulaisation and plotting offered by
pycobra.

"""

# %matplotlib inline
import numpy as np
from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.visualisation import Visualisation
from pycobra.diagnostics import Diagnostics

# setting up our random data-set
rng = np.random.RandomState(42)

# D1 = train machines; D2 = create COBRA; D3 = calibrate epsilon, alpha; D4 = testing
n_features = 2
D1, D2, D3, D4 = 200, 200, 200, 200
D = D1 + D2 + D3 + D4
X = rng.uniform(-1, 1, D * n_features).reshape(D, n_features)
# Y = np.power(X[:,1], 2) + np.power(X[:,3], 3) + np.exp(X[:,10]) 
Y = np.power(X[:,0], 2) + np.power(X[:,1], 3)

# training data-set
X_train = X[:D1 + D2]
X_test = X[D1 + D2 + D3:D1 + D2 + D3 + D4]
X_eps = X[D1 + D2:D1 + D2 + D3]
# for testing
Y_train = Y[:D1 + D2]
Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]
Y_eps = Y[D1 + D2:D1 + D2 + D3]

# set up our COBRA machine with the data
cobra = Cobra(epsilon=0.5)
cobra.fit(X_train, Y_train)


######################################################################
# Plotting COBRA
# ~~~~~~~~~~~~~~
# 
# We use the visualisation class to plot our results, and for various
# visualisations.
# 

cobra_vis = Visualisation(cobra, X_test, Y_test)

# to plot our machines, we need a linspace as input. This is the 'scale' to plot and should be the range of the results
# since our data ranges from -1 to 1 it is such - and we space it out to a hundred points
cobra_vis.plot_machines(machines=["COBRA"])

cobra_vis.plot_machines()


######################################################################
# Plots and Visualisations of Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# QQ and Boxplots!
# 

cobra_vis.QQ()

cobra_vis.boxplot()


######################################################################
# Plotting EWA!
# ~~~~~~~~~~~~~
# 
# We can use the same visualisation class for seeing how EWA works. Let's
# demonstrate this!
# 

ewa = Ewa()
ewa.set_beta(X_beta=X_eps, y_beta=Y_eps)
ewa.fit(X_train, Y_train)

ewa_vis = Visualisation(ewa, X_test, Y_test)

ewa_vis.QQ("EWA")

ewa_vis.boxplot()


######################################################################
# Plotting ClassifierCobra
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 

from sklearn import datasets
from sklearn.metrics import accuracy_score
from pycobra.classifiercobra import ClassifierCobra

bc = datasets.load_breast_cancer()
X_cc = bc.data[:-40]
y_cc = bc.target[:-40]
X_cc_test = bc.data[-40:]
y_cc_test = bc.target[-40:]

cc = ClassifierCobra()

cc.fit(X_cc, y_cc)

cc_vis = Visualisation(cc, X_cc_test, y_cc_test)

cc_vis.boxplot()


######################################################################
# Remember that all the estimators in the Pycobra package are scikit-learn
# compatible - we can also use the scikit-learn metrics and tools to
# analyse our machines!
# 

from sklearn.metrics import classification_report
print(classification_report(y_cc_test, cc.predict(X_cc_test)))


######################################################################
# Plotting COBRA colors!
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# We're now going to experiment with plotting colors and data. After we
# get information about which indices are used by which machines the best
# for a fixed epsilon (or not, we can toggle this option), we can plot the
# distribution of machines.
# 
# Why is this useful? Since we're dealing with a 2-D space now, we're
# attempting to see if there are some parts in the input space which are
# picked up by certain machines. This could lead to interesting
# experiments and
# 
# We first present a plot where the machine colors are mixed depending on
# which machines were selected; after which we plot one machine at a time.
# 

indices, MSE = cobra_vis.indice_info(X_test=X_eps[0:50], y_test=Y_eps[0:50], epsilon=0.50)

cobra_vis.color_cobra(X_test=X_eps[0:50], indice_info=indices, single=True)

cobra_vis.color_cobra(X_test=X_eps[0:50], indice_info=indices)


######################################################################
# Voronoi Tesselation
# ~~~~~~~~~~~~~~~~~~~
# 
# We present a variety of Voronoi Tesselation based plots - the purpose of
# this is to help in visualising the pattern of points which tend to be
# picked up.
# 

cobra_vis.voronoi(X_test=X_eps[0:50], indice_info=indices, single=True)

cobra_vis.voronoi(X_test=X_eps[0:50], indice_info=indices)


######################################################################
# Gradient-Colored Based Voronoi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

cobra_vis.voronoi(X_test=X_eps[0:50], indice_info=indices, MSE=MSE, gradient=True)


######################################################################
# Licensed under the MIT License - https://opensource.org/licenses/MIT
# 