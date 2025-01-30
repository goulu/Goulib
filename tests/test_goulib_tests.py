from goulib.tests import *  # a module that tests itself using itself... cool :-)


class TestPprint:
    def test_pprint(self):
        import itertools
        assert pprint(range(100)) == '0,1,2,3,4,5,6,7,8,9,...,97,98,99'
        assert pprint(range(4)) == '0,1,2,3'
        assert pprint(itertools.count()) == '0,1,2,3,4,5,6,7,8,9,...'

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
        pytest.skip("not yet implemented")  # TODO: implement


class TestPprintGen:
    def test_pprint_gen(self):
        # assert_equal(expected, pprint_gen(iterable, indices, sep))
        pytest.skip("not yet implemented")  # TODO: implement
