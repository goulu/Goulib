#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.decorators import *

class TestMemoize:
    def test_memoize(self):
        pass

class TestDebug:
    def test_debug(self):
        # assert_equal(expected, debug(func))
        raise SkipTest

class TestNodebug:
    def test_nodebug(self):
        # assert_equal(expected, nodebug(func))
        raise SkipTest

if __name__ == "__main__":
    runmodule()

