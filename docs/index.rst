Goulib
======

My Python library of useful code found and written for various projects

:author: Philippe Guglielmetti <drgoulu@gmail.com> |endorse|
:copyright: Copyright 2013 Philippe Guglielmetti
:license: LGPL (see LICENSE.TXT)

.. |endorse| image:: https://api.coderwall.com/goulu/endorsecount.png
    :target: https://coderwall.com/goulu

Modules
-------
- :mod:`datetime2` : additions to :py:mod:`datetime`
- :mod:`itertools2` : additions to :py:mod:`itertools`
- :mod:`math2` : additions to :py:mod:`math` standard library

- :mod:`interval` : operations on [x..y[ intervals
- :mod:`optim` : Optimization algorithms

  Travelling Salesman Problem (TSP) hill climbing + simulated annealing 

- :mod:`markup` : simple HTML output (branch of `markup <http://pypi.python.org/pypi/markup/>`_ )
- :mod:`nvd3` : generates Javascript charts using http://nvd3.org and http://d3js.org

  Obsolete. use `python-nvd3 <http://pypi.python.org/pypi/python-nvd3/>`_ which is derived from this module.

- :mod:`table` : Table class with CSV I/O, easy access to columns, HTML output
- :mod:`workdays` : WorkCalendar class with datetime operations on working hours

  merges and improves `BusinessHours <http://pypi.python.org/pypi/BusinessHours/>`_ and `workdays <http://pypi.python.org/pypi/workdays/>`_ packages
- :mod:`colors` : web (hex) colors dictionary and related functions

Resources
---------
:installation: "pip install Goulib"

:distribution: https://pypi.python.org/pypi/Goulib

:documentation: https://goulib.readthedocs.org/
:source: https://github.com/goulu/Goulib
:changelog: https://github.com/goulu/goulib/blob/master/CHANGES.rst

.. toctree::
   :hidden:
   :glob:
   
   modules/*

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

History
-------
.. include:: ..\CHANGES.rst