from nose.tools import assert_equal, assert_almost_equal
from nose import SkipTest

from Goulib.piecewise import *
from math import *
class TestPiecewise:
    def setup(self):
        #piecewise continuous
        self.p1=Piecewise([(4,4),(3,3.0),(1,1),(5,0)])
        self.p2=Piecewise(default=1)
        
        self.p2+=(2.5,1,6.5)
        self.p2+=(1.5,1,3.5)
        
        #boolean
        self.b1=Piecewise([(2,True)],False)
        self.b2=Piecewise([(1,True),(2,False),(3,True)],False)
        
        return 
        #simple function
        self.f=Piecewise()
        self.f+=(0,lambda x:x*x,1)
        self.f+=(0,cos,1)
        
    def test___init__(self):
        pass #tested above
    
    def test_append(self):
        pass #tested by most other tests
    
    def test_extend(self):
        pass #tested at __init__
    
    def test_index(self):
        pass #tested by most other tests
    
    def test___getitem__(self):
        pass #tested by most other tests

    def test___len__(self):
        pass #tested by most other tests
    
    def test___call__(self):
        y=[self.p1(x) for x in range(6)]
        assert_equal(y,[0,1,1,3,4,0])
        
    def test___add__(self):
        pass # += tested in setup
        y=[self.p2(x) for x in range(8)]
        assert_equal(y,[1, 1, 2, 3, 2, 2, 2, 1])
        
        p=self.p1+self.p2
        y=[p(x) for x in range(8)]
        assert_equal(y,[1, 2, 3, 6, 6, 2, 2, 1])
        
    def test___sub__(self):
        p=self.p1-self.p2
        y=[p(x) for x in range(8)]
        assert_equal(y,[-1, 0, -1, 0, 2, -2, -2, -1])
        
    def test___neg__(self):
        assert_equal(list(-self.p1),[(-inf, 0), (1, -1), (3, -3.0), (4, -4), (5, 0)])
        
    def test___mul__(self):
        p=self.p1*self.p2
        y=[p(x) for x in range(8)]
        assert_equal(y,[0, 1, 2, 9, 8, 0, 0, 0])
        
    def test___div__(self):
        p=self.p1/self.p2
        y=[p(x) for x in range(8)]
        assert_equal(y,[0.0, 1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0])
        
    def test___str__(self):
        assert_equal(str(self.p1),'[(-inf, 0), (1, 1), (3, 3.0), (4, 4), (5, 0)]')
        
    def test___iter__(self):
        pass #tested in functions below, where list(p) calls
    
    def test___invert__(self):
        assert_equal(list(~self.b2),[(-inf, True), (1, False), (2, True), (3, False)])
    
    def test___lshift__(self):
        assert_equal(list(self.b2<<2),[(-inf, False), (-1, True), (0, False), (1, True)])
    
    def test___rshift__(self):
        assert_equal(list(self.b2>>3),[(-inf, False), (4, True), (5, False), (6, True)])
        
    def test___and__(self):
        b=self.b1 & self.b2
        assert_equal(list(b),[(-inf, False), (3, True)])
        
    def test___or__(self):
        b=self.b1 | self.b2
        assert_equal(list(b),[(-inf, False), (1, True)])

    def test___xor__(self):
        b=self.b1 ^ self.b2
        assert_equal(list(b),[(-inf, False), (1, True), (3, False)])
        

    def test_applx(self):
        pass #tested in shift operators

    def test_apply(self):
        pass #tested in most operators

    def test_points(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.points(min, max, eps))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
