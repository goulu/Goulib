Goulib
======

library of useful Python code for scientific + technical applications

see the [IPython
notebook](http://nbviewer.ipython.org/github/Goulu/Goulib/blob/master/notebook.ipynb)
for an overview of features

![image](https://badge.fury.io/gh/goulu%2Fgoulib.svg%0A%20:target:%20https://badge.fury.io/gh/goulu%2Fgoulib)

![image](http://img.shields.io/badge/license-LGPL-green.svg%0A%20:target:%20https://github.com/goulu/Goulib/blob/master/LICENSE.TXT%0A%20:alt:%20License)

![image](https://badge.fury.io/py/goulib.svg%0A%20:target:%20https://pypi.python.org/pypi/Goulib/%0A%20:alt:%20Version)

![image](https://travis-ci.org/goulu/Goulib.svg?branch=master%0A%20:target:%20https://travis-ci.org/goulu/Goulib%0A%20:alt:%20Build)

![image](https://coveralls.io/repos/github/goulu/Goulib/badge.svg?branch=master%0A%20:target:%20https://coveralls.io/github/goulu/Goulib?branch=master%0A%20:alt:%20Tests)

![image](https://readthedocs.org/projects/goulib/badge/?version=latest)

> target
> :   <http://goulib.readthedocs.org/en/latest/>
>
> alt
> :   Doc
>
author
:   Philippe Guglielmetti <goulib@goulu.net>

installation
:   "pip install Goulib"

distribution
:   <https://pypi.python.org/pypi/Goulib>

documentation
:   <https://goulib.readthedocs.org/>

notebook
:   ![image](https://mybinder.org/badge_logo.svg)

    target
    :   <https://mybinder.org/v2/gh/goulu/goulib/master?filepath=notebook.ipynb>

source
:   <https://github.com/goulu/Goulib>

Modules
-------

**colors**
:   very simple RGB color management

**container**
:   sorted collection

**datetime2**
:   additions to datetime standard library

**decorators**
:   useful decorators

**drawing**
:   Read/Write and handle vector graphics in .dxf, .svg and .pdf formats

**expr**
:   simple symbolic math expressions

**geom**, **geom3d**
:   2D + 3D geometry

**graph**
:   efficient Euclidian Graphs for
    [NetworkX](http://networkx.github.io/) and related algorithms

**image**
:   image processing and conversion

**interval**
:   operations on [x..y[ intervals

**itertools2**
:   additions to itertools standard library

**markup**
:   simple HTML/XML generation (forked from
    [markup](http://pypi.python.org/pypi/markup/))

**math2**
:   additions to math standard library

**motion**
:   motion simulation (kinematics)

**optim**
:   optimization algorithms : knapsack, traveling salesman, simulated
    annealing

**piecewise**
:   piecewise-defined functions

**plot**
:   plotable rich object display on IPython notebooks

**polynomial**
:   manipulation of polynomials

**stats**
:   very basic statistics functions

**table**
:   Table class with Excel + CSV I/O, easy access to columns, HTML
    output, and much more.

**tests**
:   utilities for unit tests (using nose)

**workdays**
:   WorkCalendar class with datetime operations on working hours,
    handling holidays merges and improves
    [BusinessHours](http://pypi.python.org/pypi/BusinessHours/) and
    [workdays](http://pypi.python.org/pypi/workdays/) packages

Requirements
------------

Goulib uses lazy requirements. Many modules and functions do not require
any other packages, packages listed in requirements.txt are needed only
by some Goulib classes or functions
