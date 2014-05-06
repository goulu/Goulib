#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
useful decorators
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2014, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

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