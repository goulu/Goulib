#! /usr/bin/env python
# coding=utf-8

from setuptools import setup
import os,sys,uuid

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

from pip.req import parse_requirements
reqs = parse_requirements(
    os.path.join(os.path.dirname(__file__), "requirements.txt"),
    None,None, None, uuid.uuid1()
)

reqs = [str(ir.req) for ir in reqs]

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

    install_requires=reqs,

    test_suite="nose.collector",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
)