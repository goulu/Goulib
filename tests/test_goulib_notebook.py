from goulib.tests import *
from goulib.notebook import *

from IPython.core.interactiveshell import InteractiveShell


class TestH1:

    def test_h1(self):
        res = h1('test')
        assert res == None  # don't know how to test yet


class TestH2:

    def test_h2(self):
        res = h2('test')
        assert res == None  # don't know how to test yet


class TestH3:

    def test_h3(self):
        res = h3('test')
        assert res == None  # don't know how to test yet


class TestH4:

    def test_h4(self):
        res = h4('test')
        assert res == None  # don't know how to test yet


class TestH:

    def test_h(self):
        res = h('test')
        assert res == None  # don't know how to test yet


class TestHinfo:

    def test_hinfo(self):
        res = hinfo('test')
        assert res == None  # don't know how to test yet


class TestHsuccess:

    def test_hsuccess(self):
        res = hsuccess('test')
        assert res == None  # don't know how to test yet


class TestHwarning:

    def test_hwarning(self):
        res = hwarning('test')
        assert res == None  # don't know how to test yet


class TestHerror:

    def test_herror(self):
        res = herror('test')
        assert res == None  # don't know how to test yet


class TestHtml:

    def test_html(self):
        assert html(2) == '2'
        assert html('string') == 'string'
        assert html((1, 2, 3)) == '1 2 3'
        assert html((1, 2, 3), ',') == '(1,2,3)'
        assert html([1, 2, 3], ',') == '[1,2,3]'


class TestLatex:

    def test_latex(self):
        # assert_equal(expected, latex(obj))
        pass  # TODO: implement   # implement your test here
