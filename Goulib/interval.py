#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
operations on [x..y[ intervals
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

def in_interval(interval,x):
    ''' True if x is in interval [a,b] or [b,a] (tuple)'''
    a,b = interval[0], interval[1]
    return (a <= x <= b) or (b <= x <= a)

def intersect(t1, t2):
    ''' True if sorted tuples intervals [t1[ [t2[ intersect'''
    '''http://stackoverflow.com/questions/3721249/python-date-interval-intersection'''
    t1start, t1end = t1[0], t1[1]
    t2start, t2end = t2[0], t2[1]
    return (t1start <= t2start < t1end) or (t2start <= t1start < t2end)

def intersection(t1, t2):
    '''returns intersection between 2 intervals (tuples), 
    or (None,None) if intervals don't intersect'''
    t1start, t1end = t1[0], t1[1]
    t2start, t2end = t2[0], t2[1]
    start=max(t1start,t2start)
    end=min(t1end,t2end)
    if start>end: #no intersection
        return (None,None)
    return (start,end)

def intersectlen(t1, t2, none=0):
    '''returns len of intersection between 2 intervals (tuples), 
    or none if intervals don't intersect'''
    (start,end)=intersection(t1,t2)
    return end-start if start else none

class Interval(object):
    """
    Represents an interval. 
    Defined as half-open interval [start,end), which includes the start position but not the end.
    Start and end do not have to be numeric types. 
    
    http://code.activestate.com/recipes/576816-interval/
    alternative could be http://pypi.python.org/pypi/
    """
    
    def __init__(self, start, end):
        "Construct, start must be <= end."
        if start > end:
            raise ValueError('Start (%s) must not be greater than end (%s)' % (start, end))
        self._start = start
        self._end = end
        
    start = property(fget=lambda self: self._start, doc="The interval's start")
    end = property(fget=lambda self: self._end, doc="The interval's end")
     
    def __str__(self):
        "As string."
        return '[%s,%s)' % (self.start, self.end)
    
    def __repr__(self):
        "String representation."
        return '[%s,%s)' % (self.start, self.end)
    
    def __cmp__(self, other):
        "Compare."
        if None == other:
            return 1
        start_cmp = cmp(self.start, other.start)
        if 0 != start_cmp:
            return start_cmp
        else:
            return cmp(self.end, other.end)

    def __hash__(self):
        "Hash."
        return hash(self.start) ^ hash(self.end)
    
    def intersection(self, other):
        "Intersection. @return: None if no intersection."
        if self > other:
            other, self = self, other
        if self.end <= other.start:
            return None
        return Interval(other.start, self.end)

    def hull(self, other):
        "@return: Interval containing both self and other."
        if self > other:
            other, self = self, other
        return Interval(self.start, other.end)
    
    def overlap(self, other):
        "@return: True iff self intersects other."
        if self > other:
            other, self = self, other
        return self.end > other.start
         
    def __contains__(self, item):
        "@return: True iff item in self."
        return self.start <= item and item < self.end
         
    def contains(self,x):
        "@return: True iff 0 in self."
        return self.start <= x and x < self.end

    def subset(self, other):
        "@return: True iff self is subset of other."
        return self.start >= other.start and self.end <= other.end
         
    def proper_subset(self, other):
        "@return: True iff self is proper subset of other."
        return self.start > other.start and self.end < other.end

    def empty(self):
        "@return: True iff self is empty."
        return self.start == self.end
         
    def singleton(self):
        "@return: True iff self.end - self.start == 1."
        return self.end - self.start == 1
    
    def separation(self, other):
        "@return: The distance between self and other."
        if self > other:
            other, self = self, other
        if self.end > other.start:
            return 0
        else:
            return other.start - self.end
  
import bisect      
class Intervals(list):
    """a list of intevals kept in ascending order"""
    def __init__(self, init=[]):
        super(Intervals,self).__init__()
        self.extend(init)
        
    def extend(self,iterable):
        for i in iterable:
            self.append(i)
    
    def append(self, item):
        i=bisect.bisect_left(self,item)
        super(Intervals,self).insert(i,item)
        
    def __call__(self,x):
        """ returns list of intervals containing x"""
        return [i for i in self if i.contains(x)]
    
import unittest
class TestCase(unittest.TestCase):
    def setUp(self):
        self.i12 = Interval(1,2)
        self.i13 = Interval(1,3)
        self.i24 = Interval(2,4)
        self.intervals=Intervals([self.i24,self.i13,self.i12])
        
        
    def runTest(self):
        self.assertEqual(str(self.intervals),'[[1,2), [1,3), [2,4)]')
        

if __name__ == '__main__':
    unittest.main()
            