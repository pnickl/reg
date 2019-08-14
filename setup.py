#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: setup.py
# @Date: 2019-06-16-21-53
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from setuptools import setup

setup(name='reg',
      version='0.0.1',
      description='Regression',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['reg'],
      zip_safe=False,
)