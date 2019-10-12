#!/usr/bin/env python
# coding: utf8
from nose2.tools import assert_equal
from nose2 import SkipTest

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

    def test_plot(self):
        # plot = Plot()
        # assert_equal(expected, plot.plot(**kwargs))
        raise SkipTest 

    def test_render(self):
        # plot = Plot()
        # assert_equal(expected, plot.render(fmt, **kwargs))
        raise SkipTest 

    def test_save(self):
        # plot = Plot()
        # assert_equal(expected, plot.save(filename, **kwargs))
        raise SkipTest 

    def test_html(self):
        # plot = Plot()
        # assert_equal(expected, plot.html(**kwargs))
        raise SkipTest # implement your test here

class TestPng:
    def test_png(self):
        # assert_equal(expected, png(plotables, **kwargs))
        raise SkipTest 

class TestSvg:
    def test_svg(self):
        # assert_equal(expected, svg(plotables, **kwargs))
        raise SkipTest 

class TestSave:
    def test_save(self):
        # assert_equal(expected, save(plotables, filename, **kwargs))
        raise SkipTest 

if __name__=="__main__":
    runmodule()