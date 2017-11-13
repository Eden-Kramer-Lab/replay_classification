#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numpy >= 1.11', 'pandas >= 0.18.0', 'scipy', 'xarray',
                    'statsmodels', 'matplotlib', 'numba', 'patsy', 'seaborn',
                    'holoviews', 'bokeh']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
        name='replay_classification',
        version='0.3.0',
        license='GPL-3.0',
        description=('Non-parametric categorization of replay content from'
                     ' multiunit spiking activity'),
        author='Eric Denovellis',
        author_email='edeno@bu.edu',
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
        url='https://github.com/Eden-Kramer-Lab/replay_classification',
      )
