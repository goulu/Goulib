# -*- coding: utf-8 -*-
"""
:Id: piecewise.py
:Author: Philippe Guglielmetti <drgoulu@gmail.com>
:Copyright:  2012- , Free for non comercial use
:Description: Piecewise defined function
"""

import bisect  
import operator
from itertools import izip

class Piecewise(object):
    """
    piecewise function defined by a sorted list of (startpoint,value)
    """
    def __init__(self,init=[],default=0,start=-float('Inf')):
        #Note : started by deriving a list of (point,value), but this leads to a problem:
        # the value is taken into account in sort order by bisect
        # so instead of defining one more class with a __cmp__ method, I split both lists
        try: #copy constructor ?
            self.x=list(init.x)
            self.y=list(init.y)
        except:
            self.x=[start]
            self.y=[default]
            self.extend(init)
            
    def __call__(self,x):
        """returns value of function at point x """
        i=bisect.bisect_right(self.x,x)-1
        if i<1 : #ignore the first x value
            return self.y[0] #this is the default value right
        return self.y[i]
            
    def index(self,x,v=None):
        """finds an existing point or insert one and returns index"""
        i=bisect.bisect_left(self.x,x)
        if i<len(self) and x==self.x[i]:
            return i
        #insert either the v value, or copy the current value at x
        self.y.insert(i,v if v is not None else self(x))
        self.x.insert(i,x)
        return i
    
    def append(self, item):
        """appends a (x,y) item. In fact inserts it at correct position and returns the corresponding index"""
        return self.index(item[0],item[1])
        
    def extend(self,iterable):
        """appends an iterable of (x,y) values"""
        for p in iterable:
            self.append(p)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, i):
        return (self.x[i],self.y[i])
    
    def __iter__(self):
        return izip(self.x,self.y)
            
    def list(self):
        return [x for x in self]
    
    def __str__(self):
        return str(self.list())

    def _combine(self,other,op):
        if isinstance(other,Piecewise):
            for i,p in enumerate(other):
                try:
                    self._combine((p[0],p[1],other[i+1][0]),op)
                except:
                    self._combine((p[0],p[1]),op)
        else: #assume a triplet (start,value,end) as called above
            i=self.index(other[0])
            try:
                j=self.index(other[2])
            except:
                j=len(self)
            for k in range(i,j):
                self.y[k]=op(self.y[k],other[1])
        return self
    
    def __add__(self,other):
        return Piecewise(self)._combine(other,operator.add)
    
    def __sub__(self,other):
        return Piecewise(self)._combine(other,operator.sub)
    
    def __mul__(self,other):
        return Piecewise(self)._combine(other,operator.mul)
    
    def __div__(self,other):
        return Piecewise(self)._combine(other,operator.div)
    
    def __and__(self,other):
        return Piecewise(self)._combine(other,operator.and_)
    
    def __or__(self,other):
        return Piecewise(self)._combine(other,operator.or_)
    
    def __xor__(self,other):
        return Piecewise(self)._combine(other,operator.xor)
    
    def apply(self,f):
        """ apply a function to each piece """
        self.y=[f(v) for v in self.y]
        return self
        
    def __neg__(self):
        return Piecewise(self).apply(operator.neg)
    
    def __not__(self):
        return Piecewise(self).apply(operator.not_)

    def _opt(self):
        """removes redundant data"""
        i=1
        while i<len(self.x):
            if self.y[i]==self.y[i-1]:
                self.y.pop(i)
                self.x.pop(i)
            else:
                i+=1
        return
        
    
    def lines(self,min=0,max=None,eps=0):
        """@return x and y for a line plot"""
        self._opt()
        resx=[]
        resy=[]
        try:
            if min<self.x[1]:
                resx.append(min)
                resy.append(self(min))
        except: pass
        for i in range(1,len(self.x)):
            resx.append(self.x[i]-eps)
            resy.append(self.y[i-1])
            resx.append(self.x[i])
            resy.append(self.y[i])
        if max and max>self.x[-1]:
            resx.append(max)
            resy.append(self(max))
        return resx,resy
    
import unittest
class TestCase(unittest.TestCase):       
    def setUp(self):
        import markup
        
        self.page=markup.page()
        self.page.init(
                doctype="Content-Type: text/html; charset=utf-8\r\n\r\n<!DOCTYPE html>",
                script=["http://nvd3.org/lib/d3.v2.js","http://nvd3.org/nv.d3.js"], #must be a list to preserve order
                css=['http://nvd3.org/src/nv.d3.css']
                )
         
    def runTest(self):
        from nvd3 import LineChart
        from colors import color_range
        fig=LineChart(height=400,colors=color_range(6,'red','blue'))
        def add(p,name,min=0,max=10,disabled=False):
            print name,'=',p,'<br/>'
            x,y=p.lines(min=min,max=max)
            fig.add(x=x,y=y,name=name,disabled=disabled)
            
        p1=Piecewise([(4,4),(3,3),(1,1),(5,0)])
        self.assertEqual(str(p1),'[(-inf, 0), (1, 1), (3, 3), (4, 4), (5, 0)]')
        add(p1,'p1')
        p2=Piecewise(default=1)
        p2+=(2.5,1,6.5)
        self.assertEqual(str(p2),'[(-inf, 1), (2.5, 2), (6.5, 1)]')
        add(p2,'p2')
        add(p1+p2,'p1+p2',disabled=True)
        add(p1-p2,'p1-p2',disabled=True)
        add(p1*p2,'p1*p2',disabled=True)
        p1.apply(float) #to make division correct
        add(p1/p2,'p1/p2',disabled=True)
        self.page.add(str(fig))
        
        fig=LineChart(colors=fig.colors)
        b1=Piecewise([(2,True)],False)
        add(b1,'b1')
        b2=Piecewise([(1,True),(2,False),(3,True)],False)
        add(b2,'b2')
        add(b1 | b2,'b1 or b2',disabled=True)
        add(b1 & b2,'b1 and b2',disabled=True)
        add(b1 ^ b2,'b1 xor b2',disabled=True)
        self.page.add(str(fig))

        return
        from datetime import datetime,timedelta
        self.ptime=Piecewise([(timedelta(hours=4),4),(timedelta(hours=3),3),(timedelta(hours=1),1),(timedelta(hours=2),2),(timedelta(hours=5),0)])

    def tearDown(self):
        print self.page
        

if __name__ == '__main__':
    unittest.main()