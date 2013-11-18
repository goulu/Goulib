from nose.tools import assert_equal
from nose import SkipTest

from Goulib.expr import *
from math import *

class TestExpr:
    
    def setup(self):
        self.f1=Expr(1)
        self.fx=Expr(lambda x:x,name='x')
        self.fs=Expr(sin)
        
        self.fb1=Expr(lambda x:x>1,name='x>1')
        self.fb2=Expr(lambda x:x>2,name='x>2')
        
    def test___init__(self):
        pass #teste in setup
    
    def test___call__(self):
        assert_equal(self.f1(0),1) # constant function
        assert_equal(self.fx([-1,0,1]),[-1,0,1])
        assert_equal(self.fb1([0,1,2]),[False,False,True])

    def test___repr__(self):
        assert_equal(repr(self.f1),'1')     
        assert_equal(repr(self.fx),'x')    
        assert_equal(repr(self.fs),'sin')    
        assert_equal(repr(self.fb1),'x>1')      

    def test___add__(self):
        f=self.fx+self.f1
        assert_equal(f([-1,0,1]),[0,1,2])
        assert_equal(repr(f),'+(x,1)')     
        
    def test___neg__(self):
        assert_equal(repr(-self.f1),'-1')     
        f=-self.fx
        assert_equal(f([-1,0,1]),[1,0,-1])
        assert_equal(repr(f),'-(x)')
        
    def test___sub__(self):
        f=self.f1-self.fx
        assert_equal(f([-1,0,1]),[2,1,0])
        assert_equal(repr(f),'-(1,x)')   
        
    def test___mul__(self):
        f2=self.f1*2
        f2x=f2*self.fx
        assert_equal(f2x([-1,0,1]),[-2,0,2])
        
    def test___rmul__(self):
        f2=2*self.f1
        f2x=f2*self.fx
        assert_equal(f2x([-1,0,1]),[-2,0,2])
        
    def test___div__(self):
        f2=self.f1*2
        fx=self.fx/f2
        assert_equal(fx([-1,0,1]),[-0.5,0,0.5])
        
    def test___truediv__(self):
        pass # in fact we do only truedivs with Expr s
    
    def test_apply(self):
        f1=self.fx.apply(self.fs)
        assert_equal(f1([-1,0,1]),[sin(-1),0,sin(1)])
        f2=self.fs(self.fx)
        assert_equal(f2([-1,0,1]),[sin(-1),0,sin(1)])
    
    def test___invert__(self):
        fb=~self.fb1
        assert_equal(fb([0,1,2]),[True,True,False])

    def test___and__(self):
        fb=self.fb1 & self.fb2
        assert_equal(fb([1,2,3]),[False,False,True])

    def test___or__(self):
        fb=self.fb1 | self.fb2
        assert_equal(fb([1,2,3]),[False,True,True])

    def test___xor__(self):
        fb=self.fb1 ^ self.fb2
        assert_equal(fb([1,2,3]),[False,True,False])

    def test___cmp__(self):
        # expr = Expr(f, left, right, name)
        # assert_equal(expected, expr.__cmp__(other))
        raise SkipTest # TODO: implement your test here

    def test___lshift__(self):
        # expr = Expr(f, left, right, name)
        # assert_equal(expected, expr.__lshift__(dx))
        raise SkipTest # TODO: implement your test here

    def test___rshift__(self):
        # expr = Expr(f, left, right, name)
        # assert_equal(expected, expr.__rshift__(dx))
        raise SkipTest # TODO: implement your test here

    def test_applx(self):
        # expr = Expr(f, left, right, name)
        # assert_equal(expected, expr.applx(f, name))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
