#!/usr/bin/python
# https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py

import functools
# Decorators

#http://wiki.python.org/moin/PythonDecoratorLibrary
def memoize(obj):
    def reset():
        obj.cache = {}
    obj._reset=reset
    obj._reset()
    cache=obj.cache
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

"""
def memoize(f, maxcache=None, cache={}):
    '''Decorator to keep a cache of input/output for a given function'''
    cachelen = [0]
    def g(*args, **kwargs):
        key = (f, tuple(args), frozenset(kwargs.items()))
        if maxcache is not None and cachelen[0] >= maxcache:
            return f(*args, **kwargs)
        if key not in cache:
            cache[key] = f(*args, **kwargs)
            cachelen[0] += 1
        return cache[key]
    return g
"""

class tail_recursive(object):
    """Tail recursive decorator."""
    # Michele Simionato's version 
    CONTINUE = object()  # sentinel

    def __init__(self, func):
        self.func = func
        self.firstcall = True

    def __call__(self, *args, **kwd):
        try:
            if self.firstcall:  # start looping
                self.firstcall = False
                while True:
                    result = self.func(*args, **kwd)
                    if result is self.CONTINUE:  # update arguments
                        args, kwd = self.argskwd
                    else:  # last call
                        break
            else:  # return the arguments of the tail call
                self.argskwd = args, kwd
                return self.CONTINUE
        except:  # reset and re-raise
            self.firstcall = True
            raise
        else:  # reset and exit
            self.firstcall = True
            return result
        
class persistent(object):
    def __init__(self, it):
        self.it = it
        
    def __getitem__(self, x):
        self.it, temp = tee(self.it)
        if type(x) is slice:
            return list(islice(temp, x.start, x.stop, x.step))
        else:
            return islice(temp, x, x + 1).next()
        
    def __iter__(self):
        self.it, temp = tee(self.it)
        return temp
