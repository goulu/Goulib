from nose.tools import assert_equal
from nose import SkipTest
from Goulib.tank import *

class TestTank:

    def setup(self):
        self.tank=Tank(['hello'],len)
      
    def test___call__(self):
        pass #tested below

    def test___delitem__(self):
        pass #tested below

    def test___getitem__(self):
        pass #tested below

    def test___init__(self):
        pass #tested below

    def test___repr__(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___setitem__(self):
        pass #tested below
    
    def test_insert(self):
        pass #tested below

    def test_pop(self):
        pass #tested below

    def test_append(self):
        self.tank.append(' ')
        self.tank.append('world !')
        self.tank[0]='Bonjour'
        self.tank.insert(2,'tout le' )
        self.tank.pop()
        self.tank.append(' monde')
        assert_equal(''.join(self.tank),"Bonjour tout le monde")
        assert_equal(self.tank(),21)

    def test_count(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.count())
        raise SkipTest # TODO: implement your test here

    def test_extend(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.extend(more))
        raise SkipTest # TODO: implement your test here

    def test_index(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.index(item))
        raise SkipTest # TODO: implement your test here

    def test_remove(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.remove(item))
        raise SkipTest # TODO: implement your test here

    def test_reverse(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.reverse())
        raise SkipTest # TODO: implement your test here

    def test_sort(self):
        # tank = Tank(base, f, sum)
        # assert_equal(expected, tank.sort(**kwargs))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
