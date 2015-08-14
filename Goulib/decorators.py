#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import thread
import threading
import weakref
 
thread_pool = None
TimeoutError=multiprocessing.TimeoutError
 
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
        @functools.wraps(func)
        def __wrapper(*args, **kwargs):
            try:
                async_result = get_thread_pool().apply_async(func, args=args, kwds=kwargs)
                return async_result.get(timeout)
            except thread.error:
                return func(*args, **kwargs)
        return __wrapper
    return wrap_function