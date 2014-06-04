from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.tests import * #a module that tests itself using itself... cool :-)

class TestAssertEqual:
    def test_assert_equal(self):
        # assert_equal(expected, assert_equal(first, second, *args, **kwargs))
        raise SkipTest

class TestRunmodule:
    def test_runmodule(self):
        # assert_equal(expected, runmodule())
        raise SkipTest

if __name__=="__main__":
    runmodule()