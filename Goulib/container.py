#!/usr/bin/env python
# coding: utf8
"""
advanced containers : Record (struct), SortedCollection, and INFINITE Sequence
"""

from __future__ import division, print_function

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = ['Raymond Hettinger http://code.activestate.com/recipes/577197-sortedcollection/']
__license__ = "LGPL"

import six
from six.moves import map, filter

from bisect import bisect_left, bisect_right
from collections import OrderedDict

import operator

from itertools import count, tee, islice
from Goulib import itertools2, tests

class Record(OrderedDict):
    """mimics a Pascal record or a C struct"""
    #https://stackoverflow.com/a/5491708/1395973
    def __init__(self, *args, **kwargs):
        super(Record, self).__init__(*args, **kwargs)
        self._initialized = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not name:
            return
        if '_initialized' in self.__dict__:
            super(Record, self).__setitem__(name, value)
        else:
            super(Record, self).__setattr__(name, value)
            
    def __str__(self):
        res=['%s:%s'%(k,self[k]) for k in self]
        return '{{%s}}'%(','.join(res))
    
class Sequence(object):
    """combines a generator and a read-only list
    used for INFINITE numeric (integer) sequences
    """
    def __init__(self,iterf=None,itemf=None,containf=None,desc=''):
        """
        :param iterf: optional iterator, or a function returning an iterator
        :param itemf: optional function(i) returning the i-th element
        :param containf: optional function(n) return bool True if n belongs to Sequence
        :param desc: string description
        """
        self.name=self.__class__.__name__ #by default

        if isinstance(iterf,six.integer_types):
            self.offset=iterf
            self.iterf=None
        else:
            try: #evaluate function into iterator
                iterf=iterf()
            except Exception:
                pass
            self.offset=0
            self.iterf=iterf
        self.itemf=itemf
        if itemf and not desc:
            desc=itemf.__doc__
        self.containf=containf

        self.desc=desc

    def __repr__(self):     
        s=tests.pprint(self,[0,1,2,3,4,5,6,7,8,9]) 
        return '%s (%s ...)'%(self.name,s)
    
    def save(self, filename, comment=None, n=1000, maxtime=10):
        with open(filename,'wt') as f:
            from datetime import date
            comment = comment or "%s %s"%(self.desc,date.today())
            print('#'+comment, file=f)
            for i,v in enumerate(self):
                if i>n : break
                print(i+self.offset,v,file=f)

    def __iter__(self):
        """reset the generator
        
        :return: a tee-ed copy of iterf
        """
        if self.iterf:
            self.iterf, self.generator=tee(self.iterf)
        elif self.itemf:
            def _():
                for i in count(self.offset):
                    yield self[i]
            self.generator=_()
        else:
            def _():
                for n in count(self.offset):
                    if n in self:
                        yield n
            self.generator=_()
        return self.generator

    def __getitem__(self, i):
        if not isinstance(i,slice):
            if self.itemf :
                return self.itemf(i)
            else:
                return itertools2.index(i,self)
        else:
            return islice(self(),i.start,i.stop,i.step)

    def index(self,v):
        #assume sequence is growing
        for i,n in enumerate(self):
            if v==n: return i
            if n>v: return -1

    def __contains__(self,n):
        if self.containf:
            return self.containf(n)
        else:
            return self.index(n)>=0

    def __add__(self,other):
        if type(other) is int:
            return self.apply(
                lambda n:n+other,
                containf=lambda n:n-other in self,
                desc='%s+%d'%(self.name,other)
            )
        return self & other
                        
    def __sub__(self,other):
        if type(other) is int:
            return self+(-other)
        return self % other
            
    def __or__(self,other):
        """
        :return: Sequence with items from both operands
        """
        return Sequence(
            itertools2.merge(self,other), None,
            lambda x:x in self or x in other
        )
        
    def __and__(self,other):
        """
        :return: Sequence with items in both operands
        """
        if other.containf:
            return self.filter(other.containf)
        if self.containf:
            return other.filter(self.containf)
        raise(NotImplementedError)
    
    def __mod__(self,other):
        """
        :return: Sequence with items from left operand not in right
        """
        return Sequence(
            itertools2.diff(self.__iter__(),other.__iter__()), None,
            lambda x:x in self and x not in other
        )

    def apply(self,f,containf=None,desc=''):
        return Sequence(
            map(f,self),
            lambda i:f(self[i]),
            containf,
            desc
        )

    def filter(self,f,desc=''):
        return Sequence(
            filter(f,self),
            None,
            lambda n:f(n) and n in self,
            desc
        )

    def accumulate(self,op=operator.add,skip_first=False):
        return Sequence(itertools2.accumulate(self,op,skip_first))

    def pairwise(self,op,skip_first=False):
        return Sequence(itertools2.pairwise(self,op))

    def sort(self,key=None,buffer=100):
        return Sequence(itertools2.sorted_iterable(self, key, buffer))

    def unique(self,buffer=100):
        """ 
        :param buffer: int number of last elements found. 
        if two identical elements are separated by more than this number of elements
        in self, they might be generated twice in the resulting Sequence
        :return: Sequence made of unique elements of this one
        """
        return Sequence(itertools2.unique(self,None,buffer))