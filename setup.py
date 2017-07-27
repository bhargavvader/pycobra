#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from setuptools import setup

version = "0.2.0"
setup(name='pycobra',
      version=version,
      description='Python implementation of COBRA algorithm with regression analysis',
      author=['Bhargav Srinivasa Desikan', 'Benjamin Guedj'],
      author_email=['bhargavvader@gmail.com', 'benjamin.guedj@inria.fr'],
      url='https://github.com/bhargavvader/pycobra',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['pycobra'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib'
      ],
      test_suite='test',
      keywords=[
          'Aggregation of Predictors',
          'Regression Analysis',
          'Voronoi Tesselation',
          'Statistical Aggregation'
      ])
