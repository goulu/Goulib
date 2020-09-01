#! /usr/bin/env python
# coding=utf-8

import os, io,sys

# Import setuptools
try:
    from setuptools import setup, find_packages
    from setuptools.command.test import test as TestCommand
except ImportError as exc:
    raise RuntimeError(
        "Cannot install '{0}', setuptools is missing ({1})".format(name, exc))

# Helpers
project_root = os.path.abspath(os.path.dirname(__file__))


def srcfile(*args):
    "Helper for path building."
    return os.path.join(*((project_root,) + args))


def read(*parts):
    return open(os.path.join(os.path.dirname(__file__), *parts)).read()


def get_version():
    f = open('Goulib/__init__.py')
    try:
        for line in f:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])
    finally:
        f.close()

# Load requirements files
requirements_files = dict(
    install='requirements.txt',
    setup='setup-requirements.txt',
    test='test-requirements.txt',
)
requires = {}
for key, filename in requirements_files.items():
    requires[key] = []
    if os.path.exists(srcfile(filename)):
        with io.open(srcfile(filename), encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith('#') and ';' not in line:
                    if any(line.startswith(i) for i in ('-e', 'http://', 'https://')):
                        line = line.split('#egg=')[1]
                    elif line.startswith('http'):
                        line = line.split('#egg=')[1]
                    requires[key].append(line)

setup(
    name='Goulib',
    packages=['Goulib'],
    version=get_version(),
    description="library of useful Python code for scientific + technical applications",
    long_description=read('README.rst'),
    keywords='math, geometry, graph, optimization, drawing',
    author='Philippe Guglielmetti',
    author_email='goulib@goulu.net',
    url='http://github.com/goulu/goulib',
    license='LGPL',

    scripts=[],
    package_data={'': ['colors.csv']},

    install_requires=requires['install'],
    setup_requires=requires['setup'],
    tests_require=requires['test'],

    test_suite="nose.collector",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
)
