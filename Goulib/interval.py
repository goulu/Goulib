#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
operations on [a..b[ intervals
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

def _order(interval):
    """:return: (a,b) interval such that a<=b"""
    return (interval[0], interval[1]) if interval[0]<interval[1] else (interval[1], interval[0])

def in_interval(interval,x,closed=True):
    """:return: bool True if x is in interval [a,b] or [b,a] (tuple)"""
    a,b = _order(interval)
    return (a <= x <= b) if closed else (a <= x < b)

def intersect(t1, t2):
    """:return: bool True if intervals [t1[ [t2[ intersect"""
    '''http://stackoverflow.com/questions/3721249/python-date-interval-intersection'''
    t1start, t1end = _order(t1)
    t2start, t2end = _order(t2)
    return (t1start <= t2start < t1end) or (t2start <= t1start < t2end)

def intersection(t1, t2):
    """:return: tuple intersection between 2 intervals (tuples), 
    or None if intervals don't intersect"""
    t1start, t1end = _order(t1)
    t2start, t2end = _order(t2)
    start=max(t1start,t2start)
    end=min(t1end,t2end)
    if start>end: #no intersection
        return None
    return (start,end)

def intersectlen(t1, t2, none=0):
    '''returns len of intersection between 2 intervals (tuples), 
    or none if intervals don't intersect'''
    try:
        (start,end)=intersection(t1,t2)
        return end-start
    except:
        return none

class Interval(object):
    """
    Represents an interval. 
    Defined as half-open interval [start,end), 
    which includes the start position but not the end.
    Start and end do not have to be numeric types. 
    They might especially be time, date or timedate as used in datetime2
    
    inspired from http://code.activestate.com/recipes/576816-interval/
    alternatives could be https://pypi.python.org/pypi/interval/
     (outdated, no more doc)
     or https://pypi.python.org/pypi/pyinterval/
    """
    
    def __init__(self, start, end):
        "Construct, start must be <= end."
        self._start, self._end = _order((start,end))
        
    start = property(fget=lambda self: self._start, doc="The interval's start")
    end = property(fget=lambda self: self._end, doc="The interval's end")
     
    def __str__(self):
        "As string."
        return '[%s,%s)' % (self.start, self.end)
    
    def __repr__(self):
        "String representation."
        return '[%s,%s)' % (self.start, self.end)
    
    def __hash__(self):
        "Hash."
        return hash(self.start) ^ hash(self.end)
    
    def __lt__(self, other):
        return self.end<other.start #it has to be < even if 
    
    def __eq__(self,other):
        return self.start==other.start and self.end==other.end
    
    def intersection(self, other):
        "Intersection. :return: None if no intersection."
        if self > other:
            other, self = self, other
        if self.end <= other.start:
            return None
        return Interval(other.start, self.end)

    def hull(self, other):
        ":return: Interval containing both self and other."
        return Interval(min(self.start,other.start), max(self.end,other.end))
    
    def overlap(self, other, allow_contiguous=False):
        ":return: True iff self intersects other."
        if self > other:
            other, self = self, other
        if allow_contiguous:
            return self.end >= other.start
        else:
            return self.end > other.start
        
    def __add__(self,other):
        if self.overlap(other,True):
            return self.hull(other)
        else:
            return Intervals([self,other])
         
    def __contains__(self, x):
        ":return: True if x in self."
        return self.start <= x and x < self.end

    def subset(self, other):
        ":return: True iff self is subset of other."
        return self.start >= other.start and self.end <= other.end
         
    def proper_subset(self, other):
        ":return: True iff self is proper subset of other."
        return self.start > other.start and self.end < other.end

    def empty(self):
        ":return: True iff self is empty."
        return self.start == self.end
         
    def singleton(self):
        ":return: True iff self.end - self.start == 1."
        return self.end - self.start == 1
    
    def separation(self, other):
        ":return: The distance between self and other."
        if self > other:
            other, self = self, other
        if self.end > other.start:
            return 0
        else:
            return other.start - self.end
        
import bisect      
class Intervals(list):
    """a list of intervals kept in ascending order"""
    def __init__(self, init=[]):
        super(Intervals,self).__init__()
        self.extend(init)
        
    def extend(self,iterable):
        for i in iterable:
            self.append(i)
        return self
    
    def append(self, item):
        i=bisect.bisect_left(self,item) #item starts before self[i], but overlaps maybe with i, i+1, ... th intervals
        try:
            if self[i].overlap(item,True):
                return self.append(self.pop(i).hull(item))
        except:
            pass
        super(Intervals,self).insert(i,item)
        return self
    
    def __iadd__(self,item):
        return self.append(item)
    
    def __add__(self,item):
        return Intervals(self).append(item)
        
    def insert(self, i,x):
        raise IndexError('Intervals do not support insert. Use append or + operator.')
        
    def __call__(self,x):
        """ returns intervals containing x"""
        for interval in self:
            if x in interval:
                return interval
        return None

            