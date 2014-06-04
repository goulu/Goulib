#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
utilities for unit tests (using nose)
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014-, Philippe Guglielmetti"
__license__ = "LGPL"


import nose
import nose.tools

import collections
import itertools

def assert_equal(first, second, *args, **kwargs):
    """
    http://stackoverflow.com/a/3124155/190597 (KennyTM)
    """
    if isinstance(first,(int,bool,basestring)) and isinstance(second,(int,bool,basestring)):
        nose.tools.assert_equal(first, second, *args, **kwargs) 
    elif (isinstance(first, collections.Iterable) and isinstance(second, collections.Iterable)):
        for first_, second_ in itertools.izip_longest(
                first, second, fillvalue = object()):
            assert_equal(first_, second_, *args, **kwargs)
    else:
        try:
            nose.tools.assert_almost_equal(first, second, *args, **kwargs) 
        except TypeError: # unsupported operand type(s) for -
            nose.tools.assert_equal(first, second, *args, **kwargs) 

assert_almost_equal=assert_equal

assert_true=nose.tools.assert_true
assert_false=nose.tools.assert_false
assert_raises=nose.tools.assert_raises

raises=nose.tools.raises

SkipTest=nose.SkipTest

def runmodule():
    """ ensures stdout is printed after the tests results"""
    import sys
    from cStringIO import StringIO  
    module_name = sys.modules["__main__"].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0], module_name, '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()

runtests=runmodule