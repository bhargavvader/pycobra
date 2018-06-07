"""
pycobra and scikit-learn
========================

This notebook demonstrates pycobras integration with the scikit-learn
ecosystem. We will also give an example of pycobra's performance on some
real world data-sets.

"""

from pycobra.cobra import Cobra
from pycobra.ewa import Ewa
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation
import numpy as np
# %matplotlib inline


######################################################################
# Let's set up a synthetic data-set just to show that the COBRA estimator
# is scikit-learn compatible.
# 

# setting up our random data-set
rng = np.random.RandomState(1)

# D1 = train machines; D2 = create COBRA; D3 = calibrate epsilon, alpha; D4 = testing
n_features = 20
D1, D2, D3, D4 = 200, 200, 200, 200
D = D1 + D2 + D3 + D4
X = rng.uniform(-1, 1, D * n_features).reshape(D, n_features)
Y = np.power(X[:,1], 2) + np.power(X[:,3], 3) + np.exp(X[:,10]) 
# Y = np.power(X[:,0], 2) + np.power(X[:,1], 3)

# training data-set
X_train = X[:D1 + D2]
X_test = X[D1 + D2 + D3:D1 + D2 + D3 + D4]
X_eps = X[D1 + D2:D1 + D2 + D3]
# for testing
Y_train = Y[:D1 + D2]
Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]
Y_eps = Y[D1 + D2:D1 + D2 + D3]


######################################################################
# Similar to other scikit-learn estimators, we set up our machine by
# creating an object and then fitting it. Since we are not passing an
# Epsilon value, we pass data to find an optimal epsilon value while
# instantiating our object. The optimal epsilon is found through the
# scikit-learn ``CVGridSearch``. The ``grid_points`` parameter decides how
# many possible epsilon values must be traversed.
# 

cobra = Cobra()

cobra.set_epsilon(X_epsilon=X_eps, y_epsilon=Y_eps, grid_points=5)

cobra.epsilon

cobra.fit(X_train, Y_train)


######################################################################
# We now see if our object can fit into the scikit-learn pipeline and
# GridSearch - and it can!
# 

from sklearn.utils.estimator_checks import check_estimator
# check_estimator(Cobra) #passes


######################################################################
# Exponentially Weighted Average Aggregate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let us also demonstrate the EWA predictor. You can read more about it
# over here in the
# `paper <http://www.crest.fr/ckfinder/userfiles/files/pageperso/tsybakov/DTcolt2007.pdf>`__
# by A. Dalalyan and A. B. Tsybakov.
# 

ewa = Ewa()

ewa.set_beta(X_beta=X_eps, y_beta=Y_eps)


######################################################################
# If we fit EWA without passing beta, we perform a CV to find the optimal
# beta.
# 

ewa.fit(X_train, Y_train)

# check_estimator(Ewa) #passes


######################################################################
# EWA assigns weights to each machine based on it's MSE. We can check the
# weights of each machine with the ``plot_machine_weights`` method.
# 

ewa.plot_machine_weights()

ewa.machine_weight_


######################################################################
# Like the Cobra estimator, Ewa is also a scikit-learn compatible
# estimator. It also fits into the Visualisation class, like demonstrated
# in the
# `notebook <https://github.com/bhargavvader/pycobra/blob/master/notebooks/visualise.ipynb>`__.
# 
# Predicting?
# ~~~~~~~~~~~
# 
# Like the other scikit-learn predictors, we estimate on data by simply
# using the ``predict()`` method.
# 

query = X_test[0].reshape(1, -1)

cobra.predict(query)

ewa.predict(query)


######################################################################
# Why pycobra?
# ~~~~~~~~~~~~
# 
# There are scikit-learn estimators which already perform well in basic
# regression tasks - why use pycobra? The Cobra estimator has the
# advantage of a theoretical bound on its performance - this means it is
# supposed to perform at least as well as the estimators used to create
# it, up to a remainder term which decays to zero. The Ewa estimator also
# benefits from similar bounds.
# 
# pycobra also lets you compare the scikit-learn estimators used in the
# aggregation - unlike the ensemble methods for regression which
# scikit-learn has, pycobra's algorithms is actually built on other
# scikit-learn like estimators.
# 
# pycobra for classification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# pycobra also implements the classification algorithm as introduced by
# Mojirsheibani [1999] Combining Classifiers via Discretization, Journal
# of the American Statistical Association.
# 
# ClassifierCobra operates exactly as COBRA in the sense that data points
# are selected with respect to their closeness to the prediction of the
# new query point. Then, instead of forming a weighted average as COBRA,
# ClassifierCobra performs a majority vote to assign a label to the new
# point.
# 

from sklearn import datasets
from sklearn.metrics import accuracy_score
bc = datasets.load_breast_cancer()
X = bc.data[:-20]
y = bc.target[:-20]
X_test = bc.data[-20:]
y_test = bc.target[-20:]

from pycobra.classifiercobra import ClassifierCobra
check_estimator(ClassifierCobra)

cc = ClassifierCobra()

cc.fit(X, y)

cc.predict(X_test)


######################################################################
# Let's see how it works in a practical case.
# 

cc_diag = Diagnostics(cc, X_test, y_test)

cc_diag.load_errors()

cc_diag.machine_error


######################################################################
# Quite well!
# 


######################################################################
# Real-world datasets
# ~~~~~~~~~~~~~~~~~~~
# 
# We have demonstrated in the regression notebook how pycobra works on
# synthetic data-sets. Let's see pycobra in action on some scikit-learn
# regression datasets.
# 

diabetes = datasets.load_diabetes()

diabetes_X_train = diabetes.data[:-40]
diabetes_X_test = diabetes.data[-20:]
# part of the data to find an appropriate epsilon
diabetes_X_eps = diabetes.data[-40:-20]

diabetes_y_train = diabetes.target[:-40]
diabetes_y_test = diabetes.target[-20:]
diabetes_y_eps = diabetes.target[-40:-20]


######################################################################
# We're unaware of what epsilon value to choose for our data-sets so by
# passing ``X_eps`` and ``y_eps`` we can get an idea of what might be a
# good epsilon value.
# 

COBRA_diabetes = Cobra()
COBRA_diabetes.set_epsilon(X_epsilon=diabetes_X_eps, y_epsilon=diabetes_y_eps, grid_points=50)
COBRA_diabetes.fit(diabetes_X_train, diabetes_y_train)


######################################################################
# Predicting using the COBRA predictor is again similar to using a
# scikit-learn estimator.
# 

COBRA_diabetes.predict(diabetes_X_test)


######################################################################
# Let's compare our MSEs using the diagnostics class now.
# 

cobra_diagnostics = Diagnostics(COBRA_diabetes, diabetes_X_test, diabetes_y_test, load_MSE=True)

cobra_diagnostics.machine_MSE


######################################################################
# Let us similarily use COBRA on the Boston housing data set.
# 

boston = datasets.load_boston()

boston_X_train = boston.data[:-40]
boston_X_test = boston.data[-20:]
boston_X_eps = boston.data[-40:-20]

boston_y_train = boston.target[:-40]
boston_y_test = boston.target[-20:]
boston_y_eps = boston.target[-40:-20]

COBRA_boston = Cobra()
COBRA_boston.set_epsilon(X_epsilon=boston_X_eps, y_epsilon=boston_y_eps, grid_points=50)
COBRA_boston.fit(boston_X_train, boston_y_train)

cobra_diagnostics = Diagnostics(COBRA_boston, boston_X_test, boston_y_test, load_MSE=True)

cobra_diagnostics.machine_MSE