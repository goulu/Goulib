#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
a Tank is a container that caches sums of its content
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

# see http://docs.python.org/2/reference/datamodel.html#emulating-container-types

class Tank:
    def __init__(self, base, f,sum=sum):
        self._base=base
        self._f=f #functions that return the cached values of an item
        self.total=reduce(sum,map(f,base))
    
    def __repr__(self):
        return '%s(%s,%s)'%(self.__class__.__name__,self._base,self.total)
    
    def __call__(self):
        return self.total
    
    def count(self) : return self._base.count()
    def index(self,item) : return self._base.index(item)
    def reverse(self): return self._base.reverse()
    def sort(self,**kwargs) : self._sort(**kwargs)
    
    def __getitem__(self, key):
        return self._base[key]
    
    def _add(self,value,tot=None):
        """update total when item is added"""
        if not tot : tot=self.total
        if isinstance(value,(int,float)):
            return tot+value
        if isinstance(value,set):
            return tot|value
        return map(self._add,zip(value,tot))
    
    def _sub(self,value,tot=None,i=None):
        """update total AFTER item is removed"""
        if not tot : tot=self.total
        if isinstance(value,(int,float)):
            return tot-value
        if isinstance(value,set):
            raise(NotImplementedError)
        return map(self._sub,enumerate(zip(value,tot)))
        
    
    def __delitem__(self,key):
        self.total-=self._f(self[key])
        self._base.__delitem__(key)
    
    def __setitem__(self,key,item):
        self.total-=self._f(self[key])
        self._base.__setitem__(key,item)
        self.total+=self._f(item)
        
    def insert(self,i,item):
        self._base.insert(i,item)
        self.total+=self._f(item)
        return self
        
    def append(self,item):
        self._base.append(item)
        self.total+=self._f(item)
        return self
    
    def extend(self,more):
        for x in more:
            self.append(x)
        return self
    
    def pop(self,i=-1):
        item=self._base.pop(i)
        self.total-=self._f(item)
        return self
    
    def remove(self,item):
        self._base.remove(item)
        self.total-=self._f(item)
        return self

import unittest
class TestCase(unittest.TestCase):
    def setUp(self):
        self.tank=Tank(['hello'],len)
        
    def runTest(self):
        self.tank.append(' ')
        print self.tank
        self.tank.append('world !')
        print self.tank
        self.tank[0]='Bonjour'
        print self.tank
        self.tank.insert(2,'tout le' )
        print self.tank
        self.tank.pop()
        print self.tank
        self.tank.append('monde')
        print self.tank
        

if __name__ == '__main__':
    unittest.main()