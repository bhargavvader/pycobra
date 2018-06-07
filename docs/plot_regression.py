"""
Playing with Regression
-----------------------

This notebook will help us with testing different regression techniques,
and demonstrate the diagnostic class which can be used to find the
optimal parameters for COBRA.

So for now we will generate a random data-set and try some of the
popular regression techniques on it, after it has been loaded to COBRA.

Imports
^^^^^^^

"""

from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
import numpy as np
# %matplotlib inline


######################################################################
# Setting up data set
# ^^^^^^^^^^^^^^^^^^^
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
# Setting up COBRA
# ~~~~~~~~~~~~~~~~
# 
# Let's up our COBRA machine with the data.
# 

cobra = Cobra(random_state=0, epsilon=0.5)
cobra.fit(X_train, Y_train, default=False)


######################################################################
# When we are fitting, we initialise COBRA with an epsilon value of
# :math:`0.5` - this is because we are aware of the distribution and 0.5
# is a fair guess of what would be a "good" epsilon value, because the
# data varies from :math:`-1` to :math:`1`.
# 
# If we do not pass the :math:`\epsilon` parameter, we perform a CV on the
# training data for an optimised epsilon.
# 
# It can be noticed that the ``default`` parameter is set as false: this
# is so we can walk you through what happens when COBRA is set-up, instead
# of the deafult settings being used.
# 


######################################################################
# We're now going to split our dataset into two parts, and shuffle data
# points.
# 

cobra.split_data(D1, D1 + D2, shuffle_data=True)


######################################################################
# Let's load the default machines to COBRA.
# 

cobra.load_default()


######################################################################
# We note here that further machines can be loaded using either the
# ``loadMachine()`` and ``loadSKMachine()`` methods. The only prerequisite
# is that the machine has a valid ``predict()`` method.
# 


######################################################################
# Using COBRA's machines
# ----------------------
# 
# We've created our random dataset and now we're going to use the default
# sci-kit machines to see what the results look like.
# 

query = X_test[9].reshape(1, -1)

cobra.machines_

cobra.machines_['lasso'].predict(query)

cobra.machines_['tree'].predict(query)

cobra.machines_['ridge'].predict(query)

cobra.machines_['random_forest'].predict(query)


######################################################################
# Aggregate!
# ----------
# 
# By using the aggregate function we can combine our predictors. You can
# read about the aggregation procedure either in the original COBRA paper
# or look around in the source code for the algorithm.
# 
# We start by loading each machine's predictions now.
# 

cobra.load_machine_predictions()

cobra.predict(query)

Y_test[9]


######################################################################
# Optimizing COBRA
# ~~~~~~~~~~~~~~~~
# 
# To squeeze the best out of COBRA we make use of the COBRA diagnostics
# class. With a grid based approach to optimizing hyperparameters, we can
# find out the best epsilon value, number of machines (alpha value), and
# combination of machines.
# 


######################################################################
# Let's check the MSE for each of COBRAs machines:
# 

cobra_diagnostics = Diagnostics(cobra, X_test, Y_test, load_MSE=True)

cobra_diagnostics.machine_MSE


######################################################################
# This error is bound by the value :math:`C\mathscr{l}^{\frac{-2}{M + 2}}`
# upto a constant :math:`C`, which is problem dependant. For more details,
# we refer the user to the original
# `paper <http://www.sciencedirect.com/science/article/pii/S0047259X15000950>`__.
# 

cobra_diagnostics.error_bound


######################################################################
# Playing with Data-Splitting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# When we initially started to set up COBRA, we split our training data
# into two further parts - :math:`D_k`, and :math:`D_l`. This split was
# done 50-50, but it is upto us how we wish to do this. The following
# section will compare 20-80, 60-40, 50-50, 40-60, 80-20 and check for
# which case we get the best MSE values, for a fixed Epsilon (or use a
# grid).
# 

cobra_diagnostics.optimal_split(X_eps, Y_eps)


######################################################################
# What we saw was the default result, with the optimal split ratio and the
# corresponding MSE. We can do a further analysis here by enabling the
# info and graph options, and using more values to split on.
# 

split = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.80, 0.20), (0.90, 0.10), (0.95, 0.05)]

cobra_diagnostics.optimal_split(X_eps, Y_eps, split=split, info=True, graph=True)


######################################################################
# Alpha, Epsilon and Machines
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The following are methods to idetify the optimal epislon values, alpha
# values, and combination of machines. The grid methods allow for us to
# predict for a single point the optimal alpha/machines and epsilon
# combination.
# 
# Epsilon
# ^^^^^^^
# 
# The epsilon paramter controls how "strict" cobra should behave while
# choosing the points to aggregate.
# 

cobra_diagnostics.optimal_epsilon(X_eps, Y_eps, line_points=100)


######################################################################
# Alpha
# ^^^^^
# 
# The alpha parameter decides how many machines must a point be picked up
# by before being added to an aggregate. The default value is 4.
# 

cobra_diagnostics.optimal_alpha(X_eps, Y_eps, info=True)


######################################################################
# In this particular case, the best performance is obtained by seeking
# consensus over all 4 machines.
# 


######################################################################
# Machines
# ^^^^^^^^
# 
# Decide which subset of machines to select for the aggregate.
# 

cobra_diagnostics.optimal_machines(X_eps, Y_eps, info=True)

cobra_diagnostics.optimal_alpha_grid(X_eps[0], Y_eps[0], line_points=100)

cobra_diagnostics.optimal_machines_grid(X_eps[0], Y_eps[0], line_points=100)


######################################################################
# Increasing the number of line points helps in finding a better optimal
# value. These are the results for the same point. The MSEs are to the
# second value of the tuple.
# 
# With 10: ((('ridge', 'random\_forest', 'lasso'), 1.1063905961135443),
# 0.96254542159345469)
# 
# With 20: ((('tree', 'random\_forest'), 0.87346626008964035),
# 0.53850941611803993)
# 
# With 50: ((('ridge', 'tree'), 0.94833479666875231), 0.48256303899450931)
# 
# With 100: ((('ridge', 'tree', 'random\_forest'), 0.10058096328304948),
# 0.30285776885759158)
# 
# With 200: ((('ridge', 'tree', 'lasso'), 0.10007553130675276),
# 0.30285776885759158)
# 


######################################################################
# pycobra is Licensed under the MIT License -
# https://opensource.org/licenses/MIT
# 