#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
utilities for unit tests (using nose)
"""
from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014-, Philippe Guglielmetti"
__license__ = "LGPL"

import six

import unittest

class TestCase(unittest.TestCase):

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None, places=7, delta=None, reltol=None):
        """
        An equality assertion for ordered sequences (like lists and tuples).
        constraints on seq1,seq2 from unittest.TestCase.assertSequenceEqual are mostly removed
        
        :param seq1, seq2: sequences to compare for (quasi) equality
        :param msg: optional string message to use on failure instead of a list of differences
        :param places: int number of digits to consider in float comparisons.
                If None, enforces strict equality
        :param delta: optional float absolute tolerance value
        :param reltol: optional float relative tolerance value
        """

        if seq_type is not None:
            seq_type_name = seq_type.__name__
            if not isinstance(seq1, seq_type):
                raise self.failureException('First sequence is not a %s: %s' % (seq_type_name, str(seq1)))
            if not isinstance(seq2, seq_type):
                raise self.failureException('Second sequence is not a %s: %s' % (seq_type_name, str(seq2)))
        else:
            seq_type_name = "sequence"

        seq1_repr = str(seq1)
        seq2_repr = str(seq2)
        MAXLEN=30
        if len(seq1_repr) > MAXLEN:
            seq1_repr = seq1_repr[:MAXLEN] + '...'
        if len(seq2_repr) > MAXLEN:
            seq2_repr = seq2_repr[:MAXLEN] + '...'
        elements = (seq_type_name.capitalize(), seq1_repr, seq2_repr)
        differing = '%ss differ: %s != %s\n' % elements

        i=0
        for item1,item2 in six.moves.zip_longest(seq1,seq2):
            m=(msg if msg else differing)+'First differing element %d: %s != %s\n' %(i, item1, item2)
            self.assertEqual(item1,item2, places=places, msg=m, delta=delta, reltol=reltol)
            i+=1

    base_types=(six.integer_types,six.string_types,six.text_type,bool,set,dict)

    def assertEqual(self, first, second, places=7, msg=None, delta=None, reltol=None):
        """automatically calls assertAlmostEqual when needed
        :param first, second: objects to compare for (quasi) equality
        :param places: int number of digits to consider in float comparisons.
                        If None, forces strict equality
        :param msg: optional string error message to display in cas of failure
        :param delta: optional float absolute tolerance value
        :param reltol: optional float relative tolerance value
        """
        #inspired from http://stackoverflow.com/a/3124155/190597 (KennyTM)
        import collections
        if places is None or (isinstance(first,self.base_types) and isinstance(second,self.base_types)):
            super(TestCase,self).assertEqual(first, second,msg=msg)
        elif (isinstance(first, collections.Iterable) and isinstance(second, collections.Iterable)):
            self.assertSequenceEqual(first, second,msg=msg, places=places, delta=delta, reltol=reltol)
        elif reltol:
            ratio=first/second if second else second/first
            msg='%s != %s within %.2f%%'%(first,second,reltol*100)
            super(TestCase,self).assertAlmostEqual(ratio,1, places=None, msg=msg, delta=reltol)
        else: #float and classes
            try:
                super(TestCase,self).assertAlmostEqual(first, second, places=places, msg=msg, delta=delta)
            except TypeError: # unsupported operand type(s) for -
                super(TestCase,self).assertEqual(first, second,msg=msg)


import nose
import nose.tools

#
# Expose assert* from unittest.TestCase
# - give them pep8 style names
# (copied from nose.trivial)

import re
caps = re.compile('([A-Z])')

def pep8(name):
    return caps.sub(lambda m: '_' + m.groups()[0].lower(), name)

class Dummy(TestCase):
    def nop(self):
        pass
_t = Dummy('nop')

for at in [ at for at in dir(_t)
            if at.startswith('assert') and not '_' in at ]:
    pepd = pep8(at)
    vars()[pepd] = getattr(_t, at)
    #__all__.append(pepd)

#explicitly define the most common asserts to avoid "undefined variable" messages in IDEs
assert_true = _t.assertTrue
assert_false = _t.assertFalse
assert_equal = _t.assertEqual
assert_almost_equal = _t.assertAlmostEqual
assert_not_equal = _t.assertNotEqual
assert_raises = _t.assertRaises

del Dummy
del _t

#add other shortcuts

raises=nose.tools.raises
SkipTest=nose.SkipTest

import logging
def runmodule(redirect=True, level=logging.INFO):
    if not redirect:
        return nose.runmodule()
    
    # enable logging
    root=logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter('%(levelname)s:%(filename)s:%(funcName)s: %(message)s')
    try:
        root.handlers[0].setFormatter(fmt)
    except:
        logging.basicConfig(format=fmt)
    """ ensures stdout is printed after the tests results"""
    import sys
    from io import StringIO
    module_name = sys.modules["__main__"].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    result = nose.run(
        argv=[sys.argv[0], module_name, '-s', '--nologcapture'],
    )
    
    sys.stdout = old_stdout
    print(mystdout.getvalue())

runtests=runmodule