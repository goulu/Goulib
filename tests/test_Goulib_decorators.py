from nose.tools import assert_equal
from nose import SkipTest
from Goulib.decorators import *

class TestMemoize:
    def test_memoize(self):
        pass

class TestDebug:
    def test_debug(self):
        # assert_equal(expected, debug(func))
        raise SkipTest # TODO: implement your test here

class TestNodebug:
    def test_nodebug(self):
        # assert_equal(expected, nodebug(func))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()

