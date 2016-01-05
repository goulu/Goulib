#!/usr/bin/env python
# coding: utf8
"""
simple symbolic math expressions
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import six, operator, logging, matplotlib

from . import plot #sets matplotlib backend
import matplotlib.pyplot as plt # after import .plot

from . import itertools2

class Expr(plot.Plot):
    """
    Math expressions that can be evaluated like standard functions 
    combined using standard operators
    and plotted in Jupyter
    """
    def __init__(self,f,left=None,right=None,name=None):
        """
        :param f: function or operator, or Expr to copy construct
        :param left: Expr function parameter or left operand
        :param right: Expr function parameter or left operand
        :param name: string used in repr, preferably in LaTeX
        """
        if isinstance(f,Expr): #copy constructor
            name=f.name
            self.isconstant=f.isconstant
            self.left=f.left
            self.right=f.right
            f=f.y #to construct this object, we take only the function of the source
        if left: # f is an operator
            if not name:
                name=f.__name__
            self.left=Expr(left)
            self.right=Expr(right) if right else None
            self.isconstant=self.left.isconstant and (not self.right or self.right.isconstant)
            if self.isconstant: #simplify
                f=f(self.left.y,self.right.y) if self.right else f(self.left.y)
        else:              
            self.left=None
            self.right=None
            if hasattr(f, '__call__'):
                self.isconstant=False
                if not name:
                    name='\\%s(x)'%f.__name__ #try to build LateX name
                elif name.isalpha() and name!='x':
                    name='\\'+name
            else: #f is a constant 
                self.isconstant=True
                name=str(f)
        self.name=name
        self.y=f
        
    def __call__(self,x): 
        """evaluate the Expr at x OR compose self(x())"""
        if isinstance(x,Expr): #composition
            return self.applx(x)
        try: #is x iterable ?
            return (self(x) for x in x)
        except: pass
        if self.isconstant:
            return self.y  
        elif self.right:
            l=self.left(x)
            r=self.right(x)
            return self.y(l,r)
        if self.left:
            return self.y(self.left(x))
        else:
            return self.y(x)
        
    def __repr__(self):
        if self.isconstant:
            res=repr(self.y)
        elif self.right: #dyadic operator
            res='%s(%s,%s)'%(self.name,self.left,self.right) # no precedence worries
        elif self.left:
            res='%s(%s)'%(self.name,self.left)
        else:
            res=self.name
        return res.replace('\\','') #remove latex prefix if any
    
    def _latex(self):
        """:return: string LaTex formula"""
        if self.isconstant:
            res=repr(self.y)
        else:
            name=self.name
            left=''
            if self.left:
                left=self.left._latex()
                if self.left.left: #complex
                    left='{'+left+'}'
            right=''
            if self.right:
                right=self.right._latex()
                if self.right.left: #complex
                    right='{'+right+'}'
            if right: #dyadic operator
                res=left+name+right
            elif left: #monadic or function
                if len(name)>1:
                    res='%s(%s)'%(name,left)
                else:
                    res=name+left
            else:
                res=name
        return res
    
    def _repr_latex_(self):
        return r'$%s$'%self._latex()
    
    def latex(self):
        from IPython.display import Math
        return Math(self._latex())
    
    def _plot(self, ax, x=None, y=None, **kwargs):
        if x is None:
            x=itertools2.arange(-1,1,.1)
        x=list(x)
        if y is None:
            y=self(x)
        y=list(y)
            
        offset=kwargs.pop('offset',0) #slightly shift the points to make superimposed curves more visible
        x=[_+offset for _ in x] # listify at the same time
        y=[_+offset for _ in y] # listify at the same time
        ax.plot(x,y, **kwargs)
        return ax

    def apply(self,f,right=None,name=None):
        """function composition self o f = f(self(x)) or f(self,right)"""
        return Expr(f,self,right, name=name)
    
    def applx(self,f,name=None):
        """function composition f o self = self(f(x))"""
        res=Expr(self) #copy
        res.name=res.name.replace('(x)','')
        res.left=Expr(f,res.left,name=name)
        if res.right:
            res.right=Expr(f,res.right,name=name)
        return res
    
    def __eq__(self,other):
        if self.isconstant:
            try:
                if other.isconstant:
                    return self.y==other.y
            except:
                return self.y==other
        raise NotImplementedError #TODO: implement for general expressions...
    
    def __lt__(self,other):
        if self.isconstant:
            try:
                if other.isconstant:
                    return self.y<other.y
            except:
                return self.y<other
        raise NotImplementedError #TODO: implement for general expressions...

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
    
    def __truediv__(self,right):
        return self.apply(operator.truediv,right,'/')
    
    __div__=__truediv__
    
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

