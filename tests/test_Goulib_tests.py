from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.tests import * #a module that tests itself using itself... cool :-)

class TestAssertEqual:
    def test_assert_equal(self):
        assert_equal(0,1E-12)
        assert_equal((x**2 for x in range(2,12)),[x**2 for x in range(2,12)])

class TestRunmodule:
    def test_runmodule(self):
        pass #tested by testing ;-)

if __name__=="__main__":
    runmodule()