#!/usr/bin/python3 

from setuptools import setup, find_packages

setup(
    name='droidautoml',
    version='1.0.0',
    author='Wabbajack',
    packages=find_packages(include=['droidautoml']),
    description='DroidAutoML tests for supervised machine learning entities',
    license='MIT',
    install_requires=[
        'sklearn', 'numpy', 'pandas', 'matplotlib', 'optuna', 'mlxtend', 'termcolor', 'halo'
    ],
    test_suite='tests'
)
