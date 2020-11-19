#!/usr/bin/env python

import os
from setuptools import setup, find_packages


package_dir = os.path.dirname(__file__)
requirements_file_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

with open(requirements_file_path, 'r') as f:
    packages = [str(f) for f in f.readlines()]

setup(name='lir',
      version='0.0.5',
      description='scripts for calculating likelihood ratios',
      url='https://github.com/HolmesNL/lir/',
      author='Netherlands Forensic Institute',
      author_email='fbda@holmes.nl',
      packages=find_packages(),
      setup_requires=['nose'],
      test_suite='nose.collector',
      install_requires=packages,
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
     )
