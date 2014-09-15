#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
utilities for unit tests (using nose)
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014-, Philippe Guglielmetti"
__license__ = "LGPL"

import six

import unittest

class TestCase(unittest.TestCase):

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None, places=7, delta=None):
        """An equality assertion for ordered sequences (like lists and tuples).

        constraints on seq1,seq2 from unittest.TestCase.assertSequenceEqual are mostly removed

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
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
        for item1,item2 in zip(seq1,seq2):
            m=(msg if msg else differing)+'First differing element %d: %s != %s\n' %(i, item1, item2)
            self.assertEqual(item1,item2, places=places, msg=m, delta=delta) 
            i+=1
        
    base_types=(six.integer_types,six.string_types,six.text_type,bool)
    
    def assertEqual(self, first, second, places=7, msg=None, delta=None):
        #inspired from http://stackoverflow.com/a/3124155/190597 (KennyTM)
        import collections
        if isinstance(first,self.base_types) and isinstance(second,self.base_types):
            super(TestCase,self).assertEqual(first, second,msg=msg) 
        elif (isinstance(first, collections.Iterable) and isinstance(second, collections.Iterable)):
            self.assertSequenceEqual(first, second,msg=msg, places=places, delta=delta) 
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
    def nop():
        pass
_t = Dummy('nop')

for at in [ at for at in dir(_t)
            if at.startswith('assert') and not '_' in at ]:
    pepd = pep8(at)
    vars()[pepd] = getattr(_t, at)
    #__all__.append(pepd)

del Dummy
del _t
del pep8

#add other shortcuts

raises=nose.tools.raises
SkipTest=nose.SkipTest

def runmodule(redirect=True):
    if not redirect:
        return nose.runmodule()
    """ ensures stdout is printed after the tests results"""
    import sys
    from io import StringIO  
    module_name = sys.modules["__main__"].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0], module_name, '-s'])
    sys.stdout = old_stdout
    print(mystdout.getvalue())

runtests=runmodule