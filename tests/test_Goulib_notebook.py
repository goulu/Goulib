#!/usr/bin/env python
# coding: utf8

from nose.tools import assert_equal
from nose import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.notebook import *

from IPython.core.interactiveshell import InteractiveShell 


class TestH1:

    def test_h1(self):
        res = h1('test')
        assert_equal(res, None) # don't know how to test yet


class TestH2:

    def test_h2(self):
        res = h2('test')
        assert_equal(res, None) # don't know how to test yet


class TestH3:

    def test_h3(self):
        res = h3('test')
        assert_equal(res, None) # don't know how to test yet

class TestH4:

    def test_h4(self):
        res = h4('test')
        assert_equal(res, None) # don't know how to test yet

class TestH:

    def test_h(self):
        res = h('test')
        assert_equal(res, None) # don't know how to test yet


class TestHinfo:

    def test_hinfo(self):
        res = hinfo('test')
        assert_equal(res, None) # don't know how to test yet


class TestHsuccess:

    def test_hsuccess(self):
        res = hsuccess('test')
        assert_equal(res, None) # don't know how to test yet


class TestHwarning:

    def test_hwarning(self):
        res = hwarning('test')
        assert_equal(res, None) # don't know how to test yet


class TestHerror:

    def test_herror(self):
        res = herror('test')
        assert_equal(res, None) # don't know how to test yet


class TestHtml:

    def test_html(self):
        assert_equal(html(2),'2')
        assert_equal(html('string'),'string')
        assert_equal(html((1,2,3)),'1 2 3')
        assert_equal(html((1,2,3),','),'(1,2,3)')
        assert_equal(html([1,2,3],','),'[1,2,3]')


class TestLatex:

    def test_latex(self):
        # assert_equal(expected, latex(obj))
        raise SkipTest  # implement your test here


if __name__ == "__main__":
    runmodule()
