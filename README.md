[![Travis Status](https://travis-ci.org/bhargavvader/pycobra.svg?branch=master)](https://travis-ci.org/bhargavvader/pycobra)

## pycobra

pycobra is a python library which implements Predictor Aggregation techniques and analysis of estimators. It is scikit-learn compatible and fits into the existing scikit-learn eco-system. 

pycobra offers a python implementation of the COBRA algorithm described in the [paper](http://www.sciencedirect.com/science/article/pii/S0047259X15000950) by Biau, Fischer, Guedj and Malley [2016], COBRA: A combined regression strategy, Journal of Multivariate Analysis. 
The other algorithm implemented is the technique described in the [paper](http://www.crest.fr/ckfinder/userfiles/files/pageperso/tsybakov/DTcolt2007.pdf) by A. Dalalyan and A.B. Tsybakov, which we will refer to as the Ewa aggregate (Exponentially Weighted Average aggregate). 

pycobra also offers various visualisation and diagnostic methods built on top of matplotlib which lets the user analyse and compare different regression machines with COBRA. 
The Visualisation class also lets you use some of the tools (such as Voronoi Tesselations) on other visualisation problems, such as clustering.

### Documentation and Examples

The [notebooks](https://github.com/bhargavvader/pycobra/tree/master/notebooks) directory showcases the usage of pycobra, with examples and basic usage. 
The [documentation](https://bhargavvader.github.io) page further covers how to use pycobra.

### Installation

Run ``pip install pycobra`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``python setup.py test`` to run all tests. 

### Dependencies

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn, matplotlib