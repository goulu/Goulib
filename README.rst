Goulib
======

My Python library of useful code found and written for various projects

:author: Philippe Guglielmetti goulib@goulu.net |endorse|
:copyright: Copyright 2013 Philippe Guglielmetti
:license: LGPL (see LICENSE.TXT)

.. |endorse| image:: https://api.coderwall.com/goulu/endorsecount.png
    :target: https://coderwall.com/goulu
    
.. |travis| image:: https://travis-ci.org/goulu/Goulib.png?branch=master
    :target: https://travis-ci.org/goulu/Goulib

Modules
-------
- `datetime2` : additions to `datetime`
- `itertools2` : additions to `itertools`
- `math2` : additions to `math` standard library

- `interval` : operations on [x..y[ intervals

- `table` : Table class with Excel + CSV I/O, easy access to columns, HTML output
- `workdays` : WorkCalendar class with datetime operations on working hours

  merges and improves `BusinessHours <http://pypi.python.org/pypi/BusinessHours/>`_ and `workdays <http://pypi.python.org/pypi/workdays/>`_ packages
- `colors` : web (hex) colors dictionary and related functions

- `homcoord` : 2D homogeneous coordinates and transformations
- `dxf2img` : Rasters (simple) .dxf files to bitmap images

  (requires `dxfgrabber <http://pypi.python.org/pypi/dxfgrabber/>`_ and `pil <http://pypi.python.org/pypi/pil/>`_ )

- `optim` : Optimization algorithms

  Travelling Salesman Problem (TSP) hill climbing + simulated annealing 

- `markup` : simple HTML output (branch of `markup <http://pypi.python.org/pypi/markup/>`_ )
- `nvd3` : generates Javascript charts using http://nvd3.org and http://d3js.org

  Obsolete. use `python-nvd3 <http://pypi.python.org/pypi/python-nvd3/>`_ which is derived from this module.



Resources
---------
:installation: "pip install Goulib"

:distribution: https://pypi.python.org/pypi/Goulib

:documentation: https://goulib.readthedocs.org/

:source: https://github.com/goulu/Goulib

:changelog: https://github.com/goulu/goulib/blob/master/CHANGES.rst

:tests status: |travis|


