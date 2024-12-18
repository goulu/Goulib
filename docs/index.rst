goulib
======

library of useful Python code for scientific + technical applications

see the `IPython notebook <http://nbviewer.ipython.org/github/Goulu/goulib/blob/master/notebook.ipynb>`_ for an overview of features

.. image:: http://img.shields.io/badge/license-LGPL-green.svg
    :target: https://github.com/goulu/goulib/blob/master/LICENSE.TXT
    :alt: License
.. image:: https://badge.fury.io/py/goulib.svg
    :target: https://pypi.python.org/pypi/goulib/
    :alt: Version
.. image:: https://travis-ci.org/goulu/goulib.svg?branch=master
    :target: https://travis-ci.org/goulu/goulib
    :alt: Build
.. image:: https://coveralls.io/repos/goulu/goulib/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/goulu/goulib?branch=master
    :alt: Tests
.. image:: https://readthedocs.org/projects/goulib/badge/?version=latest
  :target: http://goulib.readthedocs.org/en/latest/
  :alt: Doc
.. image:: https://www.openhub.net/accounts/Goulu/widgets/account_tiny?format=gif
	:target: https://www.openhub.net/accounts/Goulu
.. image:: https://api.coderwall.com/goulu/endorsecount.png
    :target: https://coderwall.com/goulu
  
:author: Philippe Guglielmetti goulib@goulu.net
:installation: "pip install goulib"
:distribution: https://pypi.python.org/pypi/goulib
:documentation: https://readthedocs.org/
:notebook: http://nbviewer.ipython.org/github/Goulu/goulib/blob/master/notebook.ipynb
:source: https://github.com/goulu/goulib

Requirements
------------

goulib uses "lazy" requirements.
Many modules and functions do not require any other packages,
packages listed in requirements.txt are needed only by some classes or functions

`Sphinx <http://sphinx-doc.org/>`_ is needed to generate this documentation,
`Pythoscope <http://pythoscope.org/>`_ is used to generate nose unit tests

Modules
-------

.. currentmodule:: goulib

.. autosummary::
    :toctree: modules

    colors
    container
    datetime2
    decorators
    drawing
    expr
    geom
    geom3d
    graph
    image
    interval
    itertools2
    markup
    math2
    motion
    optim
    piecewise
    plot
    polynomial
    statemachine
    stats
    table
    tests
    workdays
   
Classes
-------

.. inheritance-diagram::
    colors
    container
    datetime2
    decorators
    drawing
    expr
    geom
    geom3d
    graph
    image
    interval
    itertools2
    math2
    motion
    optim
    piecewise
    plot
    polynomial
    statemachine
    stats
    table
    tests
    workdays
    :parts: 2

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`changes`
