#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
additions to datetime standard library
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"


from datetime import *
import datetime as dt #to distinguish from class
from interval import *

#useful constants
timedelta0=timedelta(0) 
onesecond=timedelta(seconds=1)
oneminute=timedelta(minutes=1)
onehour=timedelta(hours=1)
oneday=timedelta(days=1)
oneweek=timedelta(weeks=1)
datemin=date(year=dt.MINYEAR,month=1,day=1)

def datetimef(d,t=time(0),fmt='%Y-%m-%d'):
    '''converts something to a datetime'''
    if isinstance(d,datetime):
        return d
    if isinstance(d,basestring):
        return datetime.strptime(d,fmt)
        
    return datetime(year=d.year,month = d.month,day=d.day,
                     hour=t.hour,minute=t.minute,second=t.second)
    
def datef(d,fmt='%Y-%m-%d'):
    '''converts something to a date'''
    if isinstance(d,datetime):
        return d.date()
    if isinstance(d,date):
        return d
    if isinstance(d,basestring):
        return datetimef(d,fmt=fmt).date()
    return date(d)
    
def timef(t,fmt='%Y-%m-%d'):
    '''converts something to a time'''
    if isinstance(t,datetime):
        return t.time()
    if isinstance(t,time):
        return t
    if isinstance(t,basestring):
        return datetime(t,fmt=fmt).time()
    return time(t)

def hhmm(v):
    """ return a timedelta formated as hh:mm"""
    hours, remainder = divmod(v.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds>30: minutes+=1
    return '%d:%02d' % (hours, minutes)

def daysgen(start,length,step=oneday):
    '''returns a range of dates or datetimes'''
    i=0
    while i<length:
        i+=1
        yield start
        start=start+step
        
def days(start,length,step=oneday):
    return [x for x in daysgen(start,length,step)]

def timedelta_sum(timedeltas):
    ''' because sum(timedeltas) doesn't work...'''
    return sum(timedeltas, timedelta0)

def timedelta_div(t1,t2):
    '''divides a timedelta by a timedelta or a number. 
    should be a method of timedelta...'''
    if isinstance(t2,timedelta):
        return t1.total_seconds() / t2.total_seconds()
    else:
        return timedelta(seconds=t1.total_seconds() / t2)
    
def timedelta_mul(t1,t2):
    '''multiplies a timedelta. should be a method of timedelta...'''
    try: #timedelta is t1
        return timedelta(seconds=t1.total_seconds() * t2)
    except: #timedelta is t2
        return timedelta(seconds=t2.total_seconds() * t1)
    
def time_sub(t1,t2):
    '''substracts 2 time. should be a method of time...'''
    return datetimef(datemin,t1)-datetimef(datemin,t2)

def time_add(t,d):
    '''adds delta to time. should be a method of time...'''
    return (datetimef(datemin,t)+d).time()

def datetime_intersect(t1,t2):
    '''returns timedelta overlap between 2 intervals (tuples) of datetime'''
    a,b=intersection(t1, t2)
    if not a:return timedelta0
    return b-a

def time_intersect(t1,t2):
    '''returns timedelta overlap between 2 intervals (tuples) of time'''
    a,b=intersection(t1, t2)
    if not a:return timedelta0
    return time_sub(b,a)
        
import unittest
class TestCase(unittest.TestCase):
    def runTest(self):
        r=days(datetime.today(),21)
        self.assertEqual(len(r), 21, "incorrect days length")
        
if __name__ == '__main__':
    unittest.main()