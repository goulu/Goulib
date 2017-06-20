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
    #http://stackoverflow.com/questions/5227839/why-python-does-not-support-record-type-i-e-mutable-namedtuple
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

class SortedCollection(object):
    #
    '''Sequence sorted by a key function.

    SortedCollection() is much easier to work with than using bisect() directly.
    It supports key functions like those use in sorted(), min(), and max().
    The result of the key function call is saved so that keys can be searched
    efficiently.

    Instead of returning an insertion-point which can be hard to interpret, the
    five find-methods return a specific item in the sequence. They can scan for
    exact matches, the last item less-than-or-equal to a key, or the first item
    greater-than-or-equal to a key.

    Once found, an item's ordinal position can be located with the index() method.
    New items can be added with the insert() and insert_right() methods.
    Old items can be deleted with the remove() method.

    The usual sequence methods are provided to support indexing, slicing,
    length lookup, clearing, copying, forward and reverse iteration, contains
    checking, item counts, item removal, and a nice looking repr.

    Finding and indexing are O(log n) operations while iteration and insertion
    are O(n).  The initial sort is O(n log n).

    The key function is stored in the 'key' attibute for easy introspection or
    so that you can assign a new key function (triggering an automatic re-sort).

    In short, the class was designed to handle all of the common use cases for
    bisect but with a simpler API and support for key functions.

    >>> from pprint import pprint
    >>> from operator import itemgetter

    >>> s = SortedCollection(key=itemgetter(2))
    >>> for record in [
    ...         ('roger', 'young', 30),
    ...         ('angela', 'jones', 28),
    ...         ('bill', 'smith', 22),
    ...         ('david', 'thomas', 32)]:
    ...     s.insert(record)

    >>> pprint(list(s))         # show records sorted by age
    [('bill', 'smith', 22),
     ('angela', 'jones', 28),
     ('roger', 'young', 30),
     ('david', 'thomas', 32)]

    >>> s.find_le(29)           # find oldest person aged 29 or younger
    ('angela', 'jones', 28)
    >>> s.find_lt(28)           # find oldest person under 28
    ('bill', 'smith', 22)
    >>> s.find_gt(28)           # find youngest person over 28
    ('roger', 'young', 30)

    >>> r = s.find_ge(32)       # find youngest person aged 32 or older
    >>> s.index(r)              # get the index of their record
    3
    >>> s[3]                    # fetch the record at that index
    ('david', 'thomas', 32)

    >>> s.key = itemgetter(0)   # now sort by first name
    >>> pprint(list(s))
    [('angela', 'jones', 28),
     ('bill', 'smith', 22),
     ('david', 'thomas', 32),
     ('roger', 'young', 30)]

    '''

    def __init__(self, iterable=(), key=None):
        self._given_key = key
        key = (lambda x: x) if key is None else key
        self._items = []
        self._keys = []
        self._key = key
        for item in iterable:
            self.insert(item)

    def _getkey(self):
        return self._key

    def _setkey(self, key):
        if key is not self._key:
            self.__init__(self._items, key=key)

    def _delkey(self):
        self._setkey(None)

    key = property(_getkey, _setkey, _delkey, 'key function')

    def clear(self):
        self.__init__([], self._key)

    def copy(self):
        return self.__class__(self, self._key)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __repr__(self):
        return '%s(%r, key=%s)' % (
            self.__class__.__name__,
            self._items,
            getattr(self._given_key, '__name__', repr(self._given_key))
        )

    def __reduce__(self):
        return self.__class__, (self._items, self._given_key)

    def __contains__(self, item):
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, item):
        'Find the position of an item.  Raise ValueError if not found.'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].index(item) + i

    def count(self, item):
        'Return number of occurrences of item'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].count(item)

    def insert(self, item):
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def insert_right(self, item):
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)
        
    def pop(self,i=-1):
        del self._keys[i]
        return self._items.pop(i)

    def remove(self, item):
        'Remove first occurence of item.  Raise ValueError if not found'
        self.pop(self.index(item))

    def find(self, k):
        'Return first item with a key == k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i != len(self) and self._keys[i] == k:
            return self._items[i]
        raise ValueError('No item found with key equal to: %r' % (k,))

    def find_le(self, k):
        'Return last item with a key <= k.  Raise ValueError if not found.'
        i = bisect_right(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key at or below: %r' % (k,))

    def find_lt(self, k):
        'Return last item with a key < k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key below: %r' % (k,))

    def find_ge(self, k):
        'Return first item with a key >= equal to k.  Raise ValueError if not found'
        i = bisect_left(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key at or above: %r' % (k,))

    def find_gt(self, k):
        'Return first item with a key > k.  Raise ValueError if not found'
        i = bisect_right(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key above: %r' % (k,))
    
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