from nose2.tools import assert_equal
from nose2 import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *          # pylint: disable=wildcard-import, unused-wildcard-import
from Goulib.decorators import *     # pylint: disable=wildcard-import, unused-wildcard-import


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


class TestGetThreadPool:
    def test_get_thread_pool(self):
        # assert_equal(expected, get_thread_pool())
        raise SkipTest


class TestTimeout:
    def test_timeout(self):
        # assert_equal(expected, timeout(timeout))
        raise SkipTest 


class TestItimeout:
    def test_itimeout(self):
        # assert_equal(expected, itimeout(iterable, timeout))
        raise SkipTest 

if __name__ == "__main__":
    runmodule()

