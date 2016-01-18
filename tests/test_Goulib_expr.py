#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them

from Goulib.tests import *

from Goulib.expr import *
from math import *

import os
path=os.path.dirname(os.path.abspath(__file__))

class TestExpr:
    
    @classmethod
    def setup_class(self):
        self.f=Expr('3*x+2')
        self.f1=Expr(1)
        self.fx=Expr('x')
        self.fx2=Expr('x**2')
        self.fs=Expr('sin(x)')
        
        self.fb1=Expr('x>1')
        self.fb2=Expr('x>2')
        
        self.e1=Expr('3*x+2') #a very simple expression
        self.e1=Expr(lambda x:3*x+2) #the same as lambda
        self.e2=Expr(self.fs)
        
        self.xy=Expr('x*y')
        
        
    def test___init__(self):
        pass #teste in setup
    
    def test___call__(self):
        assert_equal(self.f1(),1) # constant function
        assert_equal(self.fx([-1,0,1]),[-1,0,1])
        assert_equal(self.fb1([0,1,2]),[False,False,True])
        
        assert_equal(self.xy(x=2,y=3),6)

    def test___str__(self):
        assert_equal(str(self.f),'3*x+2')   
        assert_equal(str(self.f1),'1')     
        assert_equal(str(self.fx),'x')    
        assert_equal(str(self.fs),'sin(x)')    
        assert_equal(str(self.fb1),'x > 1')    
        
        #test multiplication commutativity and simplification
        assert_equal(str(Expr('x*3+(a+b)')),'3*x+a+b')
        
    def test__latex(self):
        assert_equal(self.f._latex(),'3x+2')   
        assert_equal(self.f1._latex(),'1')     
        assert_equal(self.fx._latex(),'x')    
        assert_equal(self.fs._latex(),'\\sin\\left(x\\right)')    
        assert_equal(self.fb1._latex(),'x \\gtr 1')      
        assert_equal(self.fs(self.fx2)._latex(),'\\sin\\left(x^{2}\\right)') 
        

    def test___add__(self):
        f=self.fx+self.f1
        assert_equal(f([-1,0,1]),[0,1,2])
        assert_equal(str(f),'x+1')        
        
    def test___neg__(self):
        f=-self.f1
        assert_equal(str(f),'-1')        
        f=-self.fx
        assert_equal(f([-1,0,1]),[1,0,-1])
        assert_equal(str(f),'-x')
        
    def test___sub__(self):
        f=self.f1-self.fx
        assert_equal(f([-1,0,1]),[2,1,0])
        assert_equal(str(f),'1-x')   
        
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
    
    def test_applx(self):
        f=self.fs.applx(self.fx2)
        assert_equal(f(2),sin(4))
    
    def test_apply(self):
        f1=self.fx.apply(self.fs)
        assert_equal(f1([-1,0,1]),[sin(-1),0,sin(1)])
        f2=self.fs(self.fx)
        assert_equal(f2([-1,0,1]),[sin(-1),0,sin(1)])
    
    def test___not__(self):
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

    def test___lt__(self):
        assert_false(Expr(1)>Expr(2))
        assert_true(Expr(1)<2)
        
    def test___lshift__(self):
        e=self.fx<<1
        assert_equal(e(0),1)

    def test___rshift__(self):
        e=self.fx>>2
        assert_equal(e(0),-2)

    def test___eq__(self):
        assert_false(Expr(1)==Expr(2))
        assert_true(Expr(1)==1)

    def test_save(self):
        self.e2(self.e1).save(path+'/expr.png')
        self.e2(self.e1).save(path+'/expr.svg')

if __name__ == "__main__":
    runmodule()
