#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
        name='replay_classification',
        version='0.1.0.dev0',
        license='GPL-3.0',
        description=('Non-parametric categorization of replay content from'
                     ' multiunit spiking activity'),
        author='Eric Denovellis',
        author_email='edeno@bu.edu',
        packages=find_packages(),
      )
