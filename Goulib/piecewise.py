#!/usr/bin/env python
# coding: utf8
"""
piecewise-defined functions
"""

__author__ = "Philippe Guglielmetti"
__cfyright__ = "Cfyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import bisect
from . import expr, math2
    
class Piecewise(expr.Expr):
    """
    piecewise function defined by a sorted list of (startx, Expr)
    """
    def __init__(self,init=[],default=0,start=-math2.inf):
        #Note : started by deriving a list of (point,value), but this leads to a problem:
        # the value is taken into account in sort order by bisect
        # so instead of defining one more class with a __cmp__ method, I split both lists
        try: #copy constructor ?
            self.x=list(init.x)
            self.y=list(init.y)
        except:
            self.x=[]
            self.y=[]
            self.append((start,default))
            self.extend(init)

    def __call__(self,x):
        """returns value of Expr at point x """
        try: #is x iterable ?
            return [self(x) for x in x]
        except: pass
        i=bisect.bisect_right(self.x,x)-1
        if i<1 : #ignore the first x value
            return self.y[0](x) #this is the default, leftmost value
        return self.y[i](x)

    def index(self,x,v=None):
        """finds an existing point or insert one and returns its index"""
        i=bisect.bisect_left(self.x,x)
        if i<len(self) and x==self.x[i]:
            return i
        #insert either the v value, or copy the current value at x
        #note : we might have consecutive tuples with the same y value
        self.y.insert(i,expr.Expr(v) if v else self.y[i-1])
        self.x.insert(i,x)
        return i

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        """iterators through discontinuities. take the opportunity to delete redundant tuples"""
        prev=None
        i=0
        while i<len(self):
            x,y=self.x[i],self.y[i]
            if y==prev: #simplify
                self.y.pop(i)
                self.x.pop(i)
            else:
                yield x,y(x) #eval
                prev=y
                i+=1

    def append(self, item):
        """appends a (x,y) item. In fact inserts it at correct position and returns the corresponding index"""
        f=item[1]
        if not isinstance(f,expr.Expr):
            f=expr.Expr(f)
        return self.index(item[0],f)


    def extend(self,iterable):
        """appends an iterable of (x,y) values"""
        for p in iterable:
            self.append(p)

    def __getitem__(self, i):
        return (self.x[i],self.y[i])

    def __str__(self):
        return str(list(self))

    def iapply(self,f,right):
        """apply function to self"""
        if not right: #monadic . apply to each expr
            self.y=[v.apply(f) for v in self.y]
        elif isinstance(right,Piecewise): #combine each piece of right with self
            for i,p in enumerate(right):
                try:
                    self.iapply(f,(p[0],p[1],right[i+1][0]))
                except:
                    self.iapply(f,(p[0],p[1]))
        else: #assume a triplet (start,value,end) as called above
            i=self.index(right[0])
            try:
                j=self.index(right[2])
                if j<i:
                    i,j=j,i
            except:
                j=len(self)

            for k in range(i,j):
                self.y[k]=self.y[k].apply(f,right[1]) #calls Expr.apply
        return self

    def apply(self,f,right=None):
        """apply function to copy of self"""
        return Piecewise(self).iapply(f,right)

    def applx(self,f):
        """ apply a function to each x value """
        self.x=[f(x) for x in self.x]
        self.y=[y.applx(f) for y in self.y]
        return self

    def __lshift__(self,dx):
        return Piecewise(self).applx(lambda x:x-dx)

    def __rshift__(self,dx):
        return Piecewise(self).applx(lambda x:x+dx)

    def points(self,min=0, xmax=None,eps=0):
        """@return x and y for a line plot"""
        for x in self: pass #traverse to simplify through __iter__
        resx=[]
        resy=[]
        try:
            if min<self.x[1]:
                resx.append(min)
                resy.append(self(min))
        except: pass
        for i in range(1,len(self.x)):
            x=self.x[i]-eps
            resx.append(x)
            resy.append(self.y[i-1](x))
            x=self.x[i]
            resx.append(x)
            resy.append(self.y[i](x))
        if xmax and xmax>self.x[-1]:
            resx.append(xmax)
            resy.append(self(xmax))
        return resx,resy
    
    def _plot(self,ax, xmax=None,**kwargs):
        """plots function"""
        (x,y)=self.points(xmax=xmax)
        return super(Piecewise,self)._plot (ax, x, y, **kwargs)
        

