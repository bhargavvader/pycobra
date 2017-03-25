## pycobra

pycobra is a python module which implements the COBRA algorithm described by (Biau, Fischer, Guedj and Malley [2016], COBRA: A combined regression strategy, Journal of Multivariate Analysis)[http://www.sciencedirect.com/science/article/pii/S0047259X15000950]. 

The [COBRA algorithm](https://cran.r-project.org/web/packages/COBRA/index.html) is a aggregation of predictors technique which can be used to solve regression problems. pycobra also offers various visualisation and diagnostic methods built on top of matplotlib which lets the user analyse and compare different regression machines with COBRA. 

The notebooks directory showcases the usage of pycobra.

### Installation

Run ``pip install pycobra`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``python setup.py test`` to run all tests.

### Dependencies

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn, matplotlib