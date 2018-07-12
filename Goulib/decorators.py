#!/usr/bin/env python
# coding: utf8
"""
useful decorators
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = ["http://include.aorcsik.com/2014/05/28/timeout-decorator/"]
__license__ = "LGPL + MIT"

import functools

#http://wiki.python.org/moin/PythonDecoratorLibrary
def memoize(obj):
    """speed up repeated calls to a function by caching its results in a dict index by params
    :see: https://en.wikipedia.org/wiki/Memoization
    """
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


import logging

def debug(func):
    # Customize these messages
    ENTRY_MESSAGE = 'Entering {}'
    EXIT_MESSAGE = 'Exiting {}'

    @functools.wraps(func)
    def wrapper(*args, **kwds):
        logger=logging.getLogger()
        logger.info(ENTRY_MESSAGE.format(func.__name__)) 
        level=logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)
        f_result = func(*args, **kwds)
        logger.setLevel(level)
        logger.info(EXIT_MESSAGE.format(func.__name__)) 
        return f_result
    return wrapper

def nodebug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        logger=logging.getLogger()
        level=logger.getEffectiveLevel()
        logger.setLevel(logging.INFO)
        f_result = func(*args, **kwds)
        logger.setLevel(level)
        return f_result
    return wrapper

# http://include.aorcsik.com/2014/05/28/timeout-decorator/
# BUT read http://eli.thegreenplace.net/2011/08/22/how-not-to-set-a-timeout-on-a-computation-in-python

import multiprocessing
from multiprocessing.pool import ThreadPool
import six.moves._thread as thread
import threading
import weakref
 
thread_pool = None
 
def get_thread_pool():
    global thread_pool
    if thread_pool is None:
        # fix for python <2.7.2
        if not hasattr(threading.current_thread(), "_children"):
            threading.current_thread()._children = weakref.WeakKeyDictionary()
        thread_pool = ThreadPool(processes=1)
    return thread_pool
 
def timeout(timeout):
    def wrap_function(func):
        if not timeout:
            return func
        @functools.wraps(func)
        def __wrapper(*args, **kwargs):
            try:
                async_result = get_thread_pool().apply_async(func, args=args, kwds=kwargs)
                return async_result.get(timeout)
            except thread.error:
                return func(*args, **kwargs)
        return __wrapper
    return wrap_function

#https://gist.github.com/goulu/45329ef041a368a663e5
from threading import Timer
from multiprocessing import TimeoutError
    
def itimeout(iterable,timeout):
    """timeout for loops
    :param iterable: any iterable
    :param timeout: float max running time in seconds 
    :yield: items in iterator until timeout occurs
    :raise: multiprocessing.TimeoutError if timeout occured
    """
    timer=Timer(timeout,lambda:None)
    timer.start()
    for i in iterable:
        yield i
        if timer.finished.is_set(): 
            raise TimeoutError
    timer.cancel() #don't forget it, otherwise the thread never finishes...
    
# https://www.artima.com/weblogs/viewpost.jsp?thread=101605

registry = {}

class MultiMethod(object):
    def __init__(self, name):
        self.name = name
        self.typemap = {}
    def __call__(self, *args):
        types = tuple(arg.__class__ for arg in args) # a generator expression!
        function = self.typemap.get(types)
        if function is None:
            raise TypeError("no match")
        return function(*args)
    def register(self, types, function):
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function
        
def multimethod(*types):
    """
    allows to overload functions for various parameter types
    
    @multimethod(int, int)
    def foo(a, b):
        ...code for two ints...
    
    @multimethod(float, float):
    def foo(a, b):
        ...code for two floats...
    
    @multimethod(str, str):
    def foo(a, b):
        ...code for two strings...
    """
    def register(function):
        name = function.__name__
        mm = registry.get(name)
        if mm is None:
            mm = registry[name] = MultiMethod(name)
        mm.register(types, function)
        return mm
    return register