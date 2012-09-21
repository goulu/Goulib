#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
useful functions for datetime, date and time
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"


import datetime as dt

timedelta0=dt.timedelta(0) #useful constant

def datetime(d,t=dt.time(0),fmt='%Y-%m-%d'):
    '''converts something to a datetime'''
    if isinstance(d,dt.datetime):
        return d
    if isinstance(d,basestring):
        return dt.datetime.strptime(d,fmt)
        
    return dt.datetime(year=d.year,month = d.month,day=d.day,
                     hour=t.hour,minute=t.minute,second=t.second)
    
def date(d,fmt='%Y-%m-%d'):
    '''converts something to a date'''
    if isinstance(d,dt.datetime):
        return d.date()
    if isinstance(d,dt.date):
        return d
    if isinstance(d,basestring):
        return datetime(d,fmt=fmt).date()
    return dt.date(d)
    
def time(t,fmt='%Y-%m-%d'):
    '''converts something to a time'''
    if isinstance(t,dt.datetime):
        return t.time()
    if isinstance(t,time):
        return t
    if isinstance(t,basestring):
        return datetime(t,fmt=fmt).time()
    return dt.time(t)