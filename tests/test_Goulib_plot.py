#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest

from Goulib.tests import *

from Goulib.plot import *

class TestPlot:
    def test_latex(self):
        # plot = Plot()
        # assert_equal(expected, plot.latex())
        raise SkipTest 

    def test_png(self):
        # plot = Plot()
        # assert_equal(expected, plot.png())
        raise SkipTest 

    def test_svg(self):
        # plot = Plot()
        # assert_equal(expected, plot.svg())
        raise SkipTest 

if __name__=="__main__":
    runmodule()