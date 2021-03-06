"""
advanced containers : Record (struct), and INFINITE Sequence
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

from bisect import bisect_left, bisect_right
from collections import OrderedDict

import operator

from itertools import count, tee, islice, chain
from Goulib import itertools2, decorators, tests


class Record(OrderedDict):
    """mimics a Pascal record or a C struct"""

    # https://stackoverflow.com/a/5491708/1395973
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
        res = ['%s:%s' % (k, self[k]) for k in self]
        return '{{%s}}' % (','.join(res))

    
class Sequence(object):
    """combines a generator and a read-only list
    used for INFINITE numeric (integer) sequences
    """

    def __init__(self, iterf=None, itemf=None, containf=None, desc='', timeout=0):
        """
        :param iterf: optional iterator, or a function returning an iterator
        :param itemf: optional function(i) returning the i-th element
        :param containf: optional function(n) return bool True if n belongs to Sequence
        :param desc: string description
        """
        self.name = self.__class__.__name__  # by default

        if isinstance(iterf, int):
            self.offset = iterf
            self.iterf = None
        else:
            try:
                iterf = iterf()
            except TypeError as e:
                pass
            self.offset = 0
            self.iterf = iterf
        self.itemf = itemf
        if itemf and not desc:
            desc = itemf.__doc__
        self.containf = containf

        self.desc = desc
        self.timeout = timeout
        self._repr = ''
        
    def __repr__(self):     
        if not self._repr:  # cache for speed
            s = tests.pprint(self, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.001)  # must be very quick for debugger 
            self._repr = '%s = (%s)' % (self.name, s)
        return self._repr
    
    def save(self, filename, comment=None, n=1000, maxtime=10):
        with open(filename, 'wt') as f:
            from datetime import date
            comment = comment or "%s %s" % (self.desc, date.today())
            print('#' + comment, file=f)
            for i, v in enumerate(self):
                if i > n : break
                print(i + self.offset, v, file=f)

    def __iter__(self):
        """reset the generator
        
        :return: a tee-ed copy of iterf, optionally timeout decorated
        """
        if self.iterf:
            self.iterf, res = tee(self.iterf)
            if self.timeout: 
                res = decorators.itimeout(res, self.timeout)
        else:
            it = count(self.offset)
            if self.timeout: 
                it = decorators.itimeout(it, self.timeout)
            if self.itemf:
                res = map(lambda i:self[i], it)
            else:
                res = filter(lambda n:n in self, it)  # uses containf
        return res

    def __getitem__(self, i):
        if not isinstance(i, slice):
            if self.itemf :
                return self.itemf(i)
            else:
                return itertools2.index(i, self)
        else:
            return islice(self(), i.start, i.stop, i.step)

    def index(self, v):
        # assume sequence is growing
        for i, n in enumerate(self):
            if v == n: return i
            if n > v: return -1

    def __contains__(self, n):
        if self.containf:
            return self.containf(n)
        else:
            return self.index(n) >= 0

    def __add__(self, other):
        if type(other) is int:
            return self.apply(
                lambda n:n + other,
                containf=lambda n:n - other in self,
                desc='%s+%d' % (self.name, other)
            )

        def _(): 
            for (a, b) in zip(self, other): yield a + b

        return Sequence(_, lambda i:self[i] + other[i])
                        
    def __sub__(self, other):
        if type(other) is int:
            return self + (-other)

        def _(): 
            for (a, b) in zip(self, other): yield a - b

        return Sequence(_, lambda i:self[i].other[i])
    
    def __mul__(self, other):
        if type(other) is int:
            return self.apply(
                lambda n:n * other,
                containf=lambda n:other / n in self,
                desc='%d*%s' % (other, self.name)
            )

        def _(): 
            for (a, b) in zip(self, other): yield a * b

        return Sequence(_, lambda i:self[i] * other[i])
        
    def __div__(self, other):
        if type(other) is int:
            return self.apply(
                lambda n:n // other,
                containf=lambda n:other * n in self,
                desc='%s//%d' % (self.name, other)
            )

        def _(): 
            for (a, b) in zip(self, other): yield a // b

        return Sequence(_, lambda i:self[i] // other[i])
        
    def __truediv__(self, other):
        if type(other) is int:
            return self.apply(
                lambda n:n / other,
                containf=lambda n:other * n in self,
                desc='%s/%d' % (self.name, other)
            )

        def _(): 
            from fractions import Fraction
            for (a, b) in zip(self, other): yield Fraction(a, b)

        return Sequence(_, lambda i:self[i] / other[i])
            
    def __or__(self, other):
        """
        :return: Sequence with items from both (sorted) operand Sequences
        """
        return Sequence(
            itertools2.unique(itertools2.merge(self, other)),
            None,
            lambda x:x in self or x in other
        )
        
    def __and__(self, other):
        """
        :return: Sequence with items in both operands
        """
        if other.containf:
            return self.filter(other.containf)
        if self.containf:
            return other.filter(self.containf)
        raise(NotImplementedError)
    
    def __mod__(self, other):
        """
        :return: Sequence with items from left operand not in right
        """
        return Sequence(
            itertools2.diff(self.__iter__(), other.__iter__()), None,
            lambda x:x in self and x not in other
        )

    def apply(self, f, containf=None, desc=''):
        ''' function composition'''
        return Sequence(
            map(f, self),
            lambda i:f(self[i]),
            containf,
            desc
        )
        
    def __call__(self, other):
        return other.apply(self.itemf)

    def filter(self, f, desc=''):
        return Sequence(
            filter(f, self),
            None,
            lambda n:f(n) and n in self,
            desc
        )
        
    def __le__(self, other):
        return Sequence(
            itertools2.select(self, other, operator.le)
        )
        
    def __gt__(self, other):
        return Sequence(
            itertools2.select(self, other, operator.gt)
        )

    def accumulate(self, op=operator.add, init=[]):
        return Sequence(chain(init, itertools2.accumulate(self, op, False)))

    def pairwise(self, op, skip_first=False):
        return Sequence(itertools2.pairwise(self, op))

    def sort(self, key=None, buffer=100):
        return Sequence(itertools2.sorted_iterable(self, key, buffer))

    def unique(self, buffer=100):
        """ 
        :param buffer: int number of last elements found. 
        if two identical elements are separated by more than this number of elements
        in self, they might be generated twice in the resulting Sequence
        :return: Sequence made of unique elements of this one
        """
        return Sequence(itertools2.unique(self, None, buffer))
    
    def product(self, other, op=sum, buffer=100):
        """cartesian product"""
        it = itertools2.product(self, other)
        return Sequence(it).apply(op).sort(buffer=buffer)
    
