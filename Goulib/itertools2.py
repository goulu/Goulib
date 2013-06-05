#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
additions to itertools standard library
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["functional toolset from http://pyeuler.wikidot.com/toolset",
               "algos from https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py",
               "tools from http://docs.python.org/dev/py3k/library/itertools.html",
               ]
__license__ = "LGPL"

#!/usr/bin/python
from itertools import ifilter, islice, repeat, groupby
from itertools import count, imap, takewhile, tee, izip
from itertools import chain, starmap, cycle, dropwhile
import random
import logging

def take(n, iterable):
    """Take first n elements from iterable"""
    return islice(iterable, n)

def index(n, iterable):
    "Returns the nth item"
    return islice(iterable, n, n+1).next()

def first(iterable):
    """Take first element in the iterable"""
    return iterable.next()

def last(iterable):
    """Take last element in the iterable"""
    return reduce(lambda x, y: y, iterable)

def take_every(n, iterable):
    """Take an element from iterator every n elements"""
    return islice(iterable, 0, None, n)

def drop(n, iterable):
    """Drop n elements from iterable and return the rest"""
    return islice(iterable, n, None)

def ilen(it):
    """Return length exhausing an iterator"""
    return sum(1 for _ in it)

def irange(start_or_end, optional_end=None):
    """Return iterable that counts from start to end (both included)."""
    if optional_end is None:
        start, end = 0, start_or_end
    else:
        start, end = start_or_end, optional_end
    return take(max(end - start + 1, 0), count(start))

def arange(start,stop,step=1.):
    """range for floats or other types"""
    r = start
    step=abs(step)
    if stop<start : 
        while r > stop:
            yield r
            r -= step
    else:
        while r < stop:
            yield r
            r += step
        
def ilinear(start,end,n):
    """return iterator over n values linearly interpolated between (and including) start and end"""
    if isinstance(start,(int,float)):
        if start==end: #generate n times the same value for consistency
            return repeat(start,n)
        else: #make sure we generate n values including start and end
            step=float(end-start)/(n-1)
            return arange(start,end+step/2,step)
    else: #suppose start and end are tuples or lists of the same size
        res=(ilinear(s,e,n) for s,e in zip(start,end))
        return izip(*res)

def flatten(lstlsts):
    """Flatten a list of lists"""
    return (b for a in lstlsts for b in a)

def compact(it):
    """Filter None values from iterator"""
    return ifilter(bool, it)

def groups(iterable, n, step):
    """Make groups of 'n' elements from the iterable advancing
    'step' elements on each iteration"""
    itlist = tee(iterable, n)
    onestepit = izip(*(starmap(drop, enumerate(itlist))))
    return take_every(step, onestepit)

def compose(f, g):
    """Compose two functions -> compose(f, g)(x) -> f(g(x))"""
    def _wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))
    return _wrapper
  
def iterate(func, arg):
    """After Haskell's iterate: apply function repeatedly."""
    # not functional
    while 1:
        yield arg
        arg = func(arg)                

def tails(seq):
    """Get tails of a sequence: tails([1,2,3]) -> [1,2,3], [2,3], [3], []."""
    for idx in xrange(len(seq)+1):
        yield seq[idx:]
     
def ireduce(func, iterable, init=None):
    """Like reduce() but using iterators (a.k.a scanl)"""
    # not functional
    if init is None:
        iterable = iter(iterable)
        curr = iterable.next()
    else:
        curr = init
        yield init
    for x in iterable:
        curr = func(curr, x)
        yield curr

def unique(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique('AAAABBBCCDAABBB') --> A B C D
    # unique('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in iterable:
            if element not in seen:
                seen_add(element)
                yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
        
def identity(x):
    """Do nothing and return the variable untouched"""
    return x

def occurrences(it, exchange=False):
    """Return dictionary with occurrences from iterable"""
    return reduce(lambda occur, x: dict(occur, **{x: occur.get(x, 0) + 1}), it, {})

def product(*iterables, **kwargs):
    """http://stackoverflow.com/questions/12093364/cartesian-product-of-large-iterators-itertools"""
    if len(iterables) == 0:
        yield ()
    else:
        iterables = iterables * kwargs.get('repeat', 1)
        it = iterables[0]
        for item in it() if callable(it) else iter(it):
            for items in product(*iterables[1:]):
                yield (item, ) + items

# my functions added

def any(seq, pred=bool):
    "Return True if pred(x) is True for at least one element in the iterable"
    return (True in imap(pred, seq))

def all(seq, pred=bool):
    "Return True if pred(x) is True for all elements in the iterable"
    return (False not in imap(pred, seq))

def no(seq, pred=bool):
    "Returns True if pred(x) is False for every element in the iterable"
    return (True not in imap(pred, seq))

def takenth(n, iterable):
    "Returns the nth item"
    return islice(iterable, n, n+1).next()

def takeevery(n, iterable):
    """Take an element from iterator every n elements"""
    return islice(iterable, 0, None, n)

def icross(*sequences):
    """Cartesian product of sequences (recursive version)"""
    if sequences:
        for x in sequences[0]:
            for y in icross(*sequences[1:]):
                yield (x,)+y
    else: yield ()

def get_groups(iterable, n, step):
    """Make groups of 'n' elements from the iterable advancing
    'step' elements each iteration"""
    itlist = tee(iterable, n)
    onestepit = izip(*(starmap(drop, enumerate(itlist))))
    return takeevery(step, onestepit)

def quantify(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(imap(pred, iterable))
                
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def rand_seq(size):
    '''generates values in random order
    equivalent to using shuffle in random,
    without generating all values at once'''
    values=range(size)
    for i in xrange(size):
        # pick a random index into remaining values
        j=i+int(random.random()*(size-i))
        # swap the values
        values[j],values[i]=values[i],values[j]
        # return the swapped value
        yield values[i] 

def all_pairs(size):
    '''generates all i,j pairs for i,j from 0-size'''
    for i in rand_seq(size):
        for j in rand_seq(size):
            yield (i,j)
            
def split(iterable,condition):
    """@return list of elements in iterable that satisfy condition, and those that don't"""
    yes,no=[],[]
    for x in iterable:
        if condition(x): 
            yes.append(x)
        else:
            no.append(x)
    return yes,no

def next_permutation(seq, pred=cmp):
    """Like C++ std::next_permutation() but implemented as
    generator. Yields copies of seq.
    see http://blog.bjrn.se/2008/04/lexicographic-permutations-using.html"""

    def reverse(seq, start, end):
        # seq = seq[:start] + reversed(seq[start:end]) + \
        #       seq[end:]
        end -= 1
        if end <= start:
            return
        while True:
            seq[start], seq[end] = seq[end], seq[start]
            if start == end or start+1 == end:
                return
            start += 1
            end -= 1

    if not seq:
        raise StopIteration

    try:
        seq[0]
    except TypeError:
        raise TypeError("seq must allow random access.")

    first = 0
    last = len(seq)
    seq = seq[:]

    # Yield input sequence as the STL version is often
    # used inside do {} while.
    yield seq

    if last == 1:
        raise StopIteration

    while True:
        next = last - 1

        while True:
            # Step 1.
            next1 = next
            next -= 1

            if pred(seq[next], seq[next1]) < 0:
                # Step 2.
                mid = last - 1
                while not (pred(seq[next], seq[mid]) < 0):
                    mid -= 1
                seq[next], seq[mid] = seq[mid], seq[next]

                # Step 3.
                reverse(seq, next1, last)

                # Change to yield references to get rid of
                # (at worst) |seq|! copy operations.
                yield seq[:]
                break
            if next == first:
                raise StopIteration
    raise StopIteration

class iter2(object):
    """Takes in an object that is iterable.  
    http://code.activestate.com/recipes/578092-flattening-an-arbitrarily-deep-list-or-any-iterato/
    Allows for the following method calls (that should be built into iterators anyway...)
    calls:
    - append - appends another iterable onto the iterator.
    - insert - only accepts inserting at the 0 place, inserts an iterable before other iterables.
    - adding.  an iter2 object can be added to another object that is
    iterable.  i.e. iter2 + iter (not iter + iter2). 
    It's best to make all objects iter2 objects to avoid syntax errors.  :D
    """
    def __init__(self, iterable):
        self._iter = iter(iterable)
    
    def append(self, iterable):
        self._iter = chain(self._iter, iter(iterable))
        
    def insert(self, place, iterable):
        if place != 0:
            raise ValueError('Can only insert at index of 0')
        self._iter = chain(iter(iterable), self._iter)
    
    def __add__(self, iterable):
        return chain(self._iter, iter(iterable))
        
    def next(self):
        return self._iter.next()
    
    def __iter__(self):
        return self

def iflatten(iterable):
    '''flatten a list of any depth'''
    iterable = iter2(iterable)
    for e in iterable:
        if hasattr(e, '__iter__'):
            iterable.insert(0, e)
        else:
            yield e