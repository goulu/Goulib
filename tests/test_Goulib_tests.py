from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.tests import * #a module that tests itself using itself... cool :-)

class TestAssertEqual:
    def test_assert_equal(self):
        
        assert_equal(0,1E-12)
        with assert_raises(AssertionError):
            assert_equal(0,1)
            
        assert_equal('abc',u'abc')
            
        assert_equal((x**2 for x in range(12)),[x**2 for x in range(12)])
        with assert_raises(AssertionError):
            assert_equal((x**2 for x in range(12)),[x**3 for x in range(12)])
            
        assert_equal([[0,1],[2,3]],[[0,1.],[2.,3.]])
        with assert_raises(AssertionError):
            assert_equal([[0,1],[2,3]],[[0,1],[2,4]])
            """ generates this nice error msg:
            AssertionError: Sequences differ: [[0, 1], [2, 3]] != [[0, 1], [2, 4]]
            First differing element 1: [2, 3] != [2, 4]
            First differing element 1: 3 != 4
            """

class TestRunmodule:
    def test_runmodule(self):
        pass #tested by testing ;-)

class TestPep8:
    def test_pep8(self):
        pass

class TestDummy:
    def test_nop(self):
        pass

if __name__=="__main__":
    runmodule()