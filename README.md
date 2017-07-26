[![Travis Status](https://travis-ci.org/bhargavvader/pycobra.svg?branch=master)](https://travis-ci.org/bhargavvader/pycobra)

## pycobra

pycobra is a python library which serves as a toolkit for regression, prediction, and visualisation. In particular, pycobra implements aggregation (a.k.a ensemble) techniques. It is scikit-learn compatible and fits into the existing scikit-learn eco-system.

pycobra offers a python implementation of the COBRA algorithm introduced by Biau et al. (2016) for regression ([paper](http://www.sciencedirect.com/science/article/pii/S0047259X15000950)).

Another algorithm implemented is the EWA (Exponentially Weighted Aggregate) aggregation technique (among several other references, you can check the [paper](http://www.crest.fr/ckfinder/userfiles/files/pageperso/tsybakov/DTcolt2007.pdf) by Dalayan and Tsybakov (2007).

Apart from these two regression aggregation algorithms, pycobra implements a version of COBRA for classification. This procedure has been introduced by Mojirsheibani (1999) ([paper](http://www.tandfonline.com/doi/abs/10.1080/01621459.1999.10474154)). 

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


### References

- G. Biau, A. Fischer, B. Guedj and J. D. Malley (2016), COBRA: A combined regression strategy, Journal of Multivariate Analysis.
- M. Mojirsheibani (1999), Combining Classifiers via Discretization, Journal of the American Statistical Association.
- A. S. Dalalyan and A. B. Tsybakov (2007) Aggregation by exponential weighting and sharp oracle inequalities, Conference on Learning Theory.
