#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them

from goulib.tests import *  # a module that tests itself using itself... cool :-)


class TestPprint:
    def test_pprint(self):
        import itertools
        assert pprint(range(100)) == '0,1,2,3,4,5,6,7,8,9,...,97,98,99'
        assert pprint(range(4)) == '0,1,2,3'
        assert pprint(itertools.count()) == '0,1,2,3,4,5,6,7,8,9,...'


class TestAssertEqual:
    def test_assert_equal(self):

        assert 0 == 1E-12
        with assert_raises(AssertionError):
            assert 0 == 1

        assert 'abc' == u'abc'

        assert (x**2 for x in range(12)) == [x**2 for x in range(12)]
        with assert_raises(AssertionError):
            assert (x**2 for x in range(12)) == [x**3 for x in range(12)]

        assert [[0, 1], [2, 3]] == [[0, 1.], [2., 3.]]
        with assert_raises(AssertionError):
            assert [[0, 1], [2, 3]] == [[0, 1], [2, 4]]
            """ generates this nice error msg:
            AssertionError: Sequences differ: [[0, 1], [2, 3]] != [[0, 1], [2, 4]]
            First differing element 1: [2, 3] != [2, 4]
            First differing element 1: 3 != 4
            """

        assert {'a': 0, 'b': 1} == {'b': 1, 'a': 0}
        # assert_equal({'a':0,'b':1},{}) # did not fail, but how to test it ?
        assert {'a': 0, 'b': 1} != {}
        assert {'a': 'dict'} != {'another': 'dict'}


class TestAssertMatch:
    def test_assert_match(self):

        assert_match('test', 'test')
        assert_match(1548, '\d+')


class TestRunmodule:
    def test_runmodule(self):
        pass  # tested by testing ;-)


class TestPep8:
    def test_pep8(self):
        pass


class TestDummy:
    def test_nop(self):
        pass


class TestSetlog:
    def test_setlog(self):
        # assert_equal(expected, setlog(level, fmt))
        pass  # TODO: implement


class TestPprintGen:
    def test_pprint_gen(self):
        # assert_equal(expected, pprint_gen(iterable, indices, sep))
        pass  # TODO: implement


if __name__ == "__main__":
    runmodule()
