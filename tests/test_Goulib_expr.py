#!/usr/bin/env python
# coding: utf8

from __future__ import division #"true division" everywhere

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
        self.fb3=Expr('a==a') #any way to simplify it to constant True ?
        
        self.e1=Expr('3*x+2') #a very simple expression
        self.e2=Expr(self.fs)
        
        self.xy=Expr('x*y')
        self.long=Expr('(x*3+(a+b)*y)/x**(3*a*y)')
        
        self.true=Expr(True) #make sure it works
        self.false=Expr('False') #make sure it creates a bool
        
        
    def test___init__(self):
        e2=Expr(lambda x:3*x+2) # same as e1
        def f(x):return 2+x*3
        e3=Expr(f) # same as function
        fs=Expr(sin)
    
    def test___call__(self):
        assert_equal(self.f1(),1) # constant function
        assert_equal(self.fx([-1,0,1]),[-1,0,1])
        assert_equal(self.fb1([0,1,2]),[False,False,True])
        assert_equal(self.xy(x=2,y=3),6)
        #test substitution
        e=self.xy(x=2)
        assert_equal(str(e),'2y')

    def test___str__(self):
        assert_equal(str(Expr('3*5')),'3*5')
        
        assert_equal(str(self.f),'3x+2')   
        assert_equal(str(self.f1),'1')     
        assert_equal(str(self.fx),'x')    
        assert_equal(str(self.fs),'sin(x)')    
        assert_equal(str(self.fb1),'x > 1')    
        assert_equal(str(self.long),'(3x+(a+b)y)/x^(3ay)')
        
        #test multiplication commutativity and simplification
        assert_equal(str(Expr('x*3+(a+b)')),'3x+a+b')
        
    def test___repr__(self):
        assert_equal(repr(self.f),'3*x+2')   
        assert_equal(repr(self.f1),'1')     
        assert_equal(repr(self.fx),'x')    
        assert_equal(repr(self.fs),'sin(x)')    
        assert_equal(repr(self.fb1),'x > 1')    
        assert_equal(repr(self.long),'(3*x+(a+b)*y)/x**(3*a*y)')
        
        #test multiplication commutativity and simplification
        assert_equal(repr(Expr('x*3+(a+b)')),'3*x+a+b')
        
    def test_latex(self):
        assert_equal(self.f.latex(),'3x+2')   
        assert_equal(self.f1.latex(),'1')     
        assert_equal(self.fx.latex(),'x')    
        assert_equal(self.fs.latex(),'\\sin \\left(x\\right)')    
        assert_equal(self.fb1.latex(),'x \\gtr 1')      
        assert_equal(self.fs(self.fx2).latex(),'\\sin \\left(x^2\\right)') 
        assert_equal(self.long.latex(),'\\frac{3x+\\left(a+b\\right) \\cdot y}{x^{3a \\cdot y}}')
        assert_equal(Expr(sqrt(3)).latex(),'\\sqrt {3}')
        assert_equal(Expr(1./3).latex(),'\\frac{1}{3}')
        l=Expr('sqrt(x*3+(a+b)*y)/x**(3*a*y)').latex()
        assert_equal(l,'\\frac{\\sqrt {3x+\\left(a+b\\right) \\cdot y}}{x^{3a \\cdot y}}')

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
        assert_false(Expr(1)==2)
        assert_true(Expr(1)==1)
        assert_true(Expr(1)==Expr('2-1'))
        assert_equal(self.f,'3*x+2')   
        assert_equal(self.f1,'1')     
        assert_equal(self.fx,'x')    
        assert_equal(self.fs,'sin(x)')    
        assert_equal(self.fb1,'x > 1')    
        assert_equal(self.long,'(3*x+(a+b)*y)/x**(3*a*y)')

    def test_save(self):
        self.e2(self.e1).save(path+'/results/expr.png')
        self.e2(self.e1).save(path+'/results/expr.svg')

    def test___invert__(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.__invert__())
        raise SkipTest # TODO: implement your test here

    def test___truediv__(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.__truediv__(right))
        raise SkipTest # TODO: implement your test here

    def test_isconstant(self):
        # expr = Expr(f)
        # assert_equal(expected, expr.isconstant())
        raise SkipTest # TODO: implement your test here

class TestEval:
    def test_eval(self):
        # assert_equal(expected, eval(node, ctx))
        raise SkipTest # TODO: implement your test here

class TestGetFunctionSource:
    def test_get_function_source(self):
        # assert_equal(expected, get_function_source(f))
        raise SkipTest # TODO: implement your test here

class TestTextVisitor:
    def test_generic_visit(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.generic_visit(n))
        raise SkipTest # TODO: implement your test here

    def test_prec(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.prec(n))
        raise SkipTest # TODO: implement your test here

    def test_prec_BinOp(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.prec_BinOp(n))
        raise SkipTest # TODO: implement your test here

    def test_prec_UnaryOp(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.prec_UnaryOp(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_BinOp(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_BinOp(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_Call(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_Call(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_Compare(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_Compare(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_Name(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_Name(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_NameConstant(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_NameConstant(node))
        raise SkipTest # TODO: implement your test here

    def test_visit_Num(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_Num(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_UnaryOp(self):
        # text_visitor = TextVisitor()
        # assert_equal(expected, text_visitor.visit_UnaryOp(n))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # text_visitor = TextVisitor(dialect)
        raise SkipTest # TODO: implement your test here

class TestLatexVisitor:
    def test_visit_Call(self):
        # latex_visitor = LatexVisitor()
        # assert_equal(expected, latex_visitor.visit_Call(n))
        raise SkipTest # TODO: implement your test here

    def test_visit_UnaryOp(self):
        # latex_visitor = LatexVisitor()
        # assert_equal(expected, latex_visitor.visit_UnaryOp(n))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # latex_visitor = LatexVisitor()
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()
