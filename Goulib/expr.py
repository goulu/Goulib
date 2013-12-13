#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Math expressions
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import operator

class Expr(object):
    """
    Math expressions that can be evaluated like standard functions and combined using standard operators
    """

    def __init__(self,f,left=None,right=None,name=None):
        if isinstance(f,Expr): #copy constructor
            name=f.name
            self.isconstant=f.isconstant
            self.left=f.left
            self.right=f.right
            f=f.y #to construct this object, we take only the function of the source
        if left: # f is an operator
            if not name: name=f.__name__
            self.left=Expr(left)
            self.right=Expr(right) if right else None
            self.isconstant=self.left.isconstant and (not self.right or self.right.isconstant)
            if self.isconstant: #simplify
                f=f(self.left.y,self.right.y) if self.right else f(self.left.y)
        else:
            self.left=None
            self.right=None
            try: #is f a function ?
                f(1)
                self.isconstant=False
                if not name: name=f.__name__
            except: #f is a constant 
                self.isconstant=True
                name=str(f)
        self.name=name
        self.y=f
        
    def __repr__(self):
        if self.isconstant:
            return repr(self.y)
        if self.right:
            return '%s(%s,%s)'%(self.name,self.left,self.right)
        if self.left: 
            return '%s(%s)'%(self.name,self.left)
        return '%s'%self.name
        
    def __call__(self,x): 
        """evaluate the Expr at x OR compose self(x())"""
        if isinstance(x,Expr):
            return self.applx(x)
        try: #is x iterable ?
            return [self(x) for x in x]
        except: pass
        if self.isconstant:
            return self.y  
        elif self.right:
            l=self.left(x)
            r=self.right(x)
            return self.y(l,r)
        elif self.left:
            return self.y(self.left(x))
        else:
            return self.y(x)
    
    def apply(self,f,right=None,name=None):
        """function composition self o f = f(self(x)) or f(self,right)"""
        return Expr(f,self,right, name=name)
    
    def applx(self,f,name=None):
        """function composition f o self = self(self(x))"""
        res=Expr(self) #copy
        res.left=Expr(f,res.left,name=name)
        if res.right:
            res.right=Expr(f,res.right,name=name)
        return res
    
    def __cmp__(self,other):
        if self.isconstant:
            try:
                if other.isconstant:
                    return cmp(self.y,other.y)
            except:
                return cmp(self.y,other)
        raise NotImplementedError #TODO : implement for general expressions...

    def __add__(self,right):
        return self.apply(operator.add,right,'+')
    
    def __sub__(self,right):
        return self.apply(operator.sub,right,'-')
    
    def __neg__(self):
        return self.apply(operator.neg,None,'-')
    
    def __mul__(self,right):
        return self.apply(operator.mul,right,'*')
    
    def __rmul__(self,right):
        return Expr(right).apply(operator.mul,self,'*')
    
    def __div__(self,right):
        return self.apply(operator.truediv,right,'/')
    
    def __truediv__(self,right):
        return self.apply(operator.truediv,right,'/')
    
    def __invert__(self):
        return self.apply(operator.not_,None,'~')
    
    def __and__(self,right):
        return self.apply(operator.and_,right,'&')
    
    def __or__(self,right):
        return self.apply(operator.or_,right,'|')
    
    def __xor__(self,right):
        return self.apply(operator.xor,right,'^')
    
    def __lshift__(self,dx):
        return self.applx(lambda x:x+dx,name='lshift')
    
    def __rshift__(self,dx):
        return self.applx(lambda x:x-dx,name='rshift')
    
