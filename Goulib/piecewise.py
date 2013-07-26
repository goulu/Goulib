#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
piecewise-defined functions
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

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
    
    def applx(self,f):
        """ apply a function to each x value """
        self.x=[f(x) for x in self.x]
        return self
    
    def __lshift__(self,dx):
        return Piecewise(self).applx(lambda x:x-dx)
    
    def __rshift__(self,dx):
        return Piecewise(self).applx(lambda x:x+dx)
    
    def points(self,min=0,max=None,eps=0):
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
