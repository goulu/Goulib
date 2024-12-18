from nose.tools import assert_equal
from nose import SkipTest
# lines above are inserted automatically by pythoscope. Line below overrides them
from goulib.tests import *          # pylint: disable=wildcard-import, unused-wildcard-import
from goulib.decorators import *     # pylint: disable=wildcard-import, unused-wildcard-import


class TestMemoize:
    def test_memoize(self):
        pass


class TestDebug:
    def test_debug(self):
        # assert_equal(expected, debug(func))
        pass  # TODO: implement


class TestNodebug:
    def test_nodebug(self):
        # assert_equal(expected, nodebug(func))
        pass  # TODO: implement


class TestGetThreadPool:
    def test_get_thread_pool(self):
        # assert_equal(expected, get_thread_pool())
        pass  # TODO: implement


class TestTimeout:
    def test_timeout(self):
        # assert_equal(expected, timeout(timeout))
        pass  # TODO: implement


class TestItimeout:
    def test_itimeout(self):
        # assert_equal(expected, itimeout(iterable, timeout))
        pass  # TODO: implement


if __name__ == "__main__":
    runmodule()
