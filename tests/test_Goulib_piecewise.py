#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.piecewise import *
from Goulib.itertools2 import arange
from Goulib.math2 import inf

from math import *
import os
path=os.path.dirname(os.path.abspath(__file__))

class TestPiecewise:
    @classmethod
    def setup_class(self):
        #piecewise continuous
        self.p1=Piecewise([(4,4),(3,3.0),(1,1),(5,0)])
        self.p2=Piecewise(default=1)
        
        self.p2+=(2.5,1,6.5)
        self.p2+=(1.5,1,3.5)
        assert_equal(self.p2(range(8)),[1, 1, 2, 3, 2, 2, 2, 1])
        
        #boolean
        self.b1=Piecewise([(2,True)],False)
        self.b2=Piecewise([(1,True),(2,False),(3,True)],False)
        
        #simple function
        self.f=Piecewise()
        self.f+=(0,cos,1)
        self.f+=(0,lambda x:x*x,1)
        
    def test___init__(self):
        pass #tested above
    
    def test___call__(self):
        y=[self.p1(x) for x in range(6)]
        assert_equal(y,[0,1,1,3,4,0])
        return #below doesn't work yet
        y=self.f(arange(0.,2.,.1))
        assert_equal(y,[0,1,1,3,4,0])
    
    def test_points(self):
        assert_equal(self.p1.points(),([0, 1, 1, 3, 3, 4, 4, 5, 5], [0, 0, 1, 1, 3.0, 3.0, 4, 4, 0]))
        assert_equal(self.p1.points(0,5),([0, 1, 1, 3, 3, 4, 4, 5, 5], [0, 0, 1, 1, 3.0, 3.0, 4, 4, 0]))
        assert_equal(self.b2.points(), ([0, 1, 1, 2, 2, 3, 3], [False, False, True, True, False, False, True]))
    
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
        
    def test___add__(self):
        pass # += tested in setup
        
        p=self.p1+self.p2
        assert_equal(p(range(8)),[1, 2, 3, 6, 6, 2, 2, 1])
        
    def test___sub__(self):
        p=self.p1-self.p2
        y=[p(x) for x in range(8)]
        assert_equal(y,[-1, 0, -1, 0, 2, -2, -2, -1])
        
    def test___neg__(self):
        assert_equal((-self.p1),[(-inf, 0), (1, -1), (3, -3.0), (4, -4), (5, 0)])
        
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
        assert_equal(~self.b2,[(-inf, True), (1, False), (2, True), (3, False)])
    
    def test___lshift__(self):
        assert_equal(self.b2<<2,[(-inf, False), (-1, True), (0, False), (1, True)])
    
    def test___rshift__(self):
        assert_equal(self.b2>>3,[(-inf, False), (4, True), (5, False), (6, True)])
        
    def test___and__(self):
        b=self.b1 & self.b2
        assert_equal(b,[(-inf, False), (3, True)])
        
    def test___or__(self):
        b=self.b1 | self.b2
        assert_equal(b,[(-inf, False), (1, True)])

    def test___xor__(self):
        b=self.b1 ^ self.b2
        assert_equal(b,[(-inf, False), (1, True), (3, False)])
        

    def test_applx(self):
        pass #tested in shift operators

    def test_apply(self):
        pass #tested in most operators


    def test_iapply(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.iapply(f, right, name))
        raise SkipTest 
    
    def test_save(self):
        self.p2.save(path+'/piecewise.p2.png',xmax=7,ylim=(-1,5))
        
    def test_svg(self):
        svg=self.p2._repr_svg_(xmax=7,ylim=(-1,5)) # return IPython object
        with open(path+'/piecewise.p2.svg','wb') as f:
            f.write(svg.encode('utf-8'))

if __name__ == "__main__":
    runmodule()
