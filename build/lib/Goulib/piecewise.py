# -*- coding: utf-8 -*-
"""
:Id: piecewise.py
:Author: Philippe Guglielmetti <drgoulu@gmail.com>
:Copyright:  2012- , Free for non comercial use
:Description: Piecewise defined function
"""

import bisect  
class Piecewise(object):
    """
    sorted list of (startpoint,value)
    inspired from faces.resource.ResourceCalendar
    """
    def __init__(self,init=[],default=0):
        #Note : started by deriving a list of (point,value), but this leads to a problem:
        # the value is taken into account in sort order by bisect
        # so instead of defining one more class with a __cmp__ method, I split the two lists
        self._index=[]
        self._value=[]
        self._default=default #"leftmost" value, before the first tuple
        self.extend(init)
        
    def extend(self,iterable):
        """appends a list of (startpoint,value)"""
        for p in iterable:
            self.append(p)
            
    def __call__(self,x):
        """ returns value of function at x """
        i=bisect.bisect_right(self._index,x)-1
        if i<0 :
            return self._default
        return self._value[i]
            
    def _point(self,x,v=None):
        """finds an existing point or insert one and returns index"""
        i=bisect.bisect_left(self._index,x)
        if i<len(self) and x==self._index[i]:
            return i
        self._value.insert(i,v if v is not None else self(x))
        self._index.insert(i,x)
        return i
    
    def append(self, item):
        i=self._point(item[0],item[1])
        
    def __len__(self):
        return len(self._index)
        
    def __getitem__(self, i):
        return (self._index[i],self._value[i])
    
    def __iter__(self):
        for i,id in enumerate(self._index):
            yield (id,self._value[i])
            
    def xy(self):
        return self._index,self._value
            
    def list(self):
        return [x for x in self]
    
    def __str__(self):
        return str(self.list())
        
    def __add__(self,pieces):
        if isinstance(pieces,Piecewise):
            self._default+=pieces._default
            for i,p in enumerate(pieces):
                self.__add__((p[0],p[1],self[i+1][0] if i+1<len(self) else None))
        else: #assume a triplet (start,end,value) as called above
            i=self._point(pieces[0])
            j=self._point(pieces[2]) if pieces[2] else len(self)
            for k in range(i,j):
                self._value[k]+=pieces[1]
        return self
    
import unittest
class TestCase(unittest.TestCase):
    def setUp(self):
        self.p1=Piecewise([(4,4),(3,3),(1,1),(2,2),(5,0)])
        self.p2=Piecewise()
        from datetime import datetime,timedelta
        self.ptime=Piecewise([(timedelta(hours=4),4),(timedelta(hours=3),3),(timedelta(hours=1),1),(timedelta(hours=2),2),(timedelta(hours=5),0)])
        
        
    def runTest(self):
        self.assertEqual(str(self.p1),'[(1, 1), (2, 2), (3, 3), (4, 4), (5, 0)]')
        self.assertEqual(map(self.p1,[0,1,2.1,4.9,5]),[0, 1, 2, 4, 0])
        self.p2+(-2,1,None)+(4.5,2,None)+(1.5,3,4.5)
        print self.p1,'+',self.p2,"="
        print self.p1+self.p2
        
        

if __name__ == '__main__':
    unittest.main()
            