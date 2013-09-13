#!/usr/bin/python
# -*- coding: utf-8 -*-
# https://github.com/tokland/pyeuler/blob/master/pyeuler/toolset.py

import functools
# Decorators

"""
#doen't work as expected...
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

