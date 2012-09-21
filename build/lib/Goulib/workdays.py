#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
date and time operations on restricted calendars
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["https://github.com/ogt/workdays",
               "http://groups.google.com/group/comp.lang.python/browse_thread/thread/ddd39a02644540b7"
               ]
__license__ = "LGPL"

from datetime import date, datetime, timedelta, time

from datetime2 import timedelta0

# Define the weekday mnemonics to match the date.weekday function
(MON, TUE, WED, THU, FRI, SAT, SUN) = range(7)

def daysrange(start,length):
    '''returns a range of days'''
    return [start + timedelta(days=x) for x in range(0,length) ]

def in_interval(interval,x):
    ''' True if x is in interval defined by a tuple (unsorted).'''
    a,b = interval[0], interval[1]
    #return cmp(a,x) * cmp(x,b) > 0
    return a <= x and x <= b or b <= x and x <= a

def _datetime(d,t=time(0)):
    '''converts a date and an optional time to a datetime'''
    if isinstance(d,datetime):
        return d
    else:
        return datetime(year=d.year,month = d.month,day=d.day,
                     hour=t.hour,minute=t.minute,second=t.second)
    
def _date(d):
    '''converts a datetime to a date'''
    if isinstance(d,datetime):
        return d.date()
    else:
        return d
    
def _time(t):
    '''converts something to a time'''
    if isinstance(t,datetime):
        return t.time()
    elif isinstance(t,time):
        return t
    else:
        return time(t)

def timedelta_div(t1,t2):
    '''divides a timedelta by a timedelta or a number. 
    should be a method of timedelta...'''
    if isinstance(t2,timedelta):
        return t1.total_seconds() / t2.total_seconds()
    else:
        return timedelta(seconds=t1.total_seconds() / t2)
    
def timedelta_mul(t1,t2):
    '''multiplies a timedelta. should be a method of timedelta...'''
    if isinstance(t1,timedelta):
        return timedelta(seconds=t1.total_seconds() * t2)
    else:
        return timedelta(seconds=t2.total_seconds() * t1)
    
def time_sub(t1,t2):
    '''substracts 2 time. should be a method of time...'''
    day=date.today()
    return _datetime(day,t1)-_datetime(day,t2)

def time_add(t,d):
    '''adds delta to time. should be a method of time...'''
    day=date.today()
    return (_datetime(day,t)+d).time()

def interval_intersect(t1, t2):
    '''http://stackoverflow.com/questions/3721249/python-date-interval-intersection'''
    t1start, t1end = t1[0], t1[1]
    t2start, t2end = t2[0], t2[1]
    return (t1start <= t2start <= t1end) or (t2start <= t1start <= t2end)

def interval_intersection(t1, t2):
    '''returns overlap between 2 intervals (tuples), 
    or (None,None) if intervals don't intersect'''
    t1start, t1end = t1[0], t1[1]
    t2start, t2end = t2[0], t2[1]
    start=max(t1start,t2start)
    end=min(t1end,t2end)
    if start>end: #no intersection
        return None,None
    return start,end

def interval_intersect_datetime(t1,t2):
    '''returns timedelta overlap between 2 intervals (tuples) of datetime'''
    a,b=interval_intersection(t1, t2)
    if not a:return timedelta0
    return b-a

def interval_intersect_time(t1,t2):
    '''returns timedelta overlap between 2 intervals (tuples) of time'''
    a,b=interval_intersection(t1, t2)
    if not a:return timedelta0
    return time_sub(b,a)

class WorkCalendar:
    '''
    inspired by http://pypi.python.org/pypi/BusinessHours
    '''
 
    def __init__(self,worktiming,holidays=[],weekends=[SAT,SUN]):
        self.start = _time(worktiming[0])
        self.end = _time(worktiming[1])
        self.weekends=weekends
        self.holidays=set(holidays)
        self.delta=timedelta(minutes=(self.end.hour-self.start.hour)*60+
                             (self.end.minute-self.start.minute))
        self._oneday=timedelta(days=1) #useful "constant"
    
    def addholidays(self,days):
        '''add day(s) to to known holidays. dates with year==1 apply every year
        note : holidays set may contain weekends too.'''
        if isinstance(days,list):
            self.holidays.update(days)
        elif isinstance(days,date):
            self.holidays.add(days)
        else:
            assert(False)
        
    def isworkday(self,day):
        if day.weekday() in self.weekends: return False
        if _date(day) in self.holidays: return False
        if date(year=4,month=day.month,day=day.day) in self.holidays: return False
        return True
    
    def isworktime(self,time):
        if not self.isworkday(time): return False
        return in_interval(self.workdatetime(time),time)
    
    def nextworkday(self,day):
        res=day
        while True:
            res=res+self._oneday
            if self.isworkday(res): break
        return res
    
    def prevworkday(self,day):
        res=day
        while True:
            res=res-self._oneday
            if self.isworkday(res): break
        return res
    
    def range(self,start,end):
        """range of workdays between start (included) and end (not included)"""
        if start>end:
            return self.range(end,start)
        res=[]
        day=start
        if not self.isworkday(day):
            day=self.nextworkday(start)
        while day<end:
            res.append(day)
            day=self.nextworkday(day)
        return res
    
    def workdays(self,start_date,ndays):
        """list of ndays workdays from start"""
        day=start_date
        res=[day]
        while ndays>0:
            day=self.nextworkday(day)
            ndays=ndays-1
            res.append(day)
        while ndays<0:
            day=self.prevworkday(day)
            ndays=ndays+1
            res.insert(0,day)
        return res
    
    def workday(self,start_date,ndays):
        '''Same as Excel WORKDAY function.
        Returns a date that is the indicated number of working days before or after the starting date. 
        Working days exclude weekends and any dates identified as holidays.
        Use WORKDAY to exclude weekends or holidays when you calculate invoice due dates, 
        expected delivery times, or the number of days of work performed.
        '''
        if ndays>0:
            return self.workdays(start_date,ndays)[-1]
        else:
            return self.workdays(start_date,ndays)[0]
    
    def cast(self,time,retro=False):
        '''force time to be in workhours'''
        if self.isworktime(time): 
            return time #ok
        if retro:
            if not self.isworkday(time) or time.time()<self.start:
                return _datetime(self.prevworkday(time.date()),self.end)
            #only remaining case is time>self.end on a work day
            return _datetime(time.date(),self.end)
        else:
            if not self.isworkday(time) or time.time()>self.end:
                return _datetime(self.nextworkday(time.date()),self.start)
            #only remaining case is time<self.start on a work day
            return _datetime(time.date(),self.start)
    
    def worktime(self,day):
        '''interval of time worked a given day'''
        if not self.isworkday(day): return None
        return (self.start,self.end)
    
    def workdatetime(self,day):
        '''interval of datetime worked a given day'''
        if not self.isworkday(day): return None
        day=_date(day)
        return (_datetime(day,self.start),_datetime(day,self.end))
    
    def diff(self,t1,t2):
        '''timedelta worktime between t1 and t2 (= t2-t1)'''
        t1=_datetime(t1,self.start)
        t2=_datetime(t2,self.start)
        if t1>t2: return -self.diff(t2,t1)
        fulldays=max(0,self.networkdays(t1, t2)-2)
        res=timedelta_mul(fulldays,self.delta)
        w1=self.workdatetime(t1)
        if w1: res+=interval_intersect_datetime(w1,(t1,t2))
        w2=self.workdatetime(t2)
        if w2: res+=interval_intersect_datetime(w2,(t1,t2))
        return res
        
    def gethours(self,t1,t2):
        return self.diff(t1,t2).total_seconds()/3600.
    
    def _add(self,start,t):
        '''adds t work time (positive or negative) to start time'''
        start=_datetime(start,self.start)
        assert self.isworktime(start), "start time not in worktime"
        days=timedelta_div(t,self.delta)
        res=start
        while days>=1:
            res=self.nextworkday(res)
            days=days-1
        while days<=-1:
            res=self.prevworkday(res)
            days=days+1
        
        remaining=timedelta_mul(self.delta,days) #less than one day of work
        day=res.date()
        start=_datetime(day,self.start)
        end=_datetime(day,self.end)
        if (res+remaining)<start: # skip to previous day
            remaining=(res+remaining)-start #in full time
            res=_datetime(self.prevworkday(day),self.end)
        if (res+remaining)>end: # skip to next day
            remaining=(res+remaining)-end #in full time
            res=_datetime(self.nextworkday(day),self.start)
        return res+remaining
    
    def _sub(self,start,t):
        return self._add(start,-t)

    def networkdays(self,start_date, end_date):
        '''Same as Excel NETWORKDAYS function.
        Returns the number of whole working days between 
        start_date and end_date (inclusive of both start_date and end_date). 
        Working days exclude weekends and any dates identified in holidays. 
        Use NETWORKDAYS to calculate employee benefits that accrue 
        based on the number of days worked during a specific term'''
        if end_date<start_date:
            return -self.networkdays(end_date,start_date)
        end_date=_date(end_date)
        i=_date(start_date)
        res=0
        while i<=end_date:
            if self.isworkday(i):res+=1
            i=_date(i+self._oneday)
        return res
    
''' a 24/24 7/7 calendar is useful'''
FullTime=WorkCalendar([time.min,time.max],holidays=[],weekends=[]) 
FullTime.delta=timedelta(hours=24) #make it perfect

''' compatibility with http://pypi.python.org/pypi/BusinessHours'''
   
def workday(start_date,ndays,holidays=[]):
    '''Same as Excel WORKDAY function.
    Returns a date that is the indicated number of working days before or after the starting date. 
    Working days exclude weekends and any dates identified as holidays.
    Use WORKDAY to exclude weekends or holidays when you calculate invoice due dates, 
    expected delivery times, or the number of days of work performed.
    '''
    return WorkCalendar([8,16],holidays).workday(start_date,ndays)

def networkdays(start_date, end_date,holidays=[]):
    '''Same as Excel NETWORKDAYS function.
    Returns the number of whole working days between 
    start_date and end_date (inclusive of both start_date and end_date). 
    Working days exclude weekends and any dates identified in holidays. 
    Use NETWORKDAYS to calculate employee benefits that accrue 
    based on the number of days worked during a specific term'''
    return WorkCalendar([8,16],holidays).networkdays(start_date,end_date)
                        
import unittest
class TestCase(unittest.TestCase):
    def runTest(self):
        cal=WorkCalendar([8,16])
        date1=date(2011,12,16) # a FRI close to year end and holidays
        assert cal.networkdays(date1,date1) == 1, 'networkdays'
        assert cal.networkdays(date1,date(2011,12,17)) == 1, 'networkdays'
        assert cal.networkdays(date1,date(2011,12,18)) == 1, 'networkdays'
        assert cal.networkdays(date1,date(2011,12,19)) == 2, 'networkdays'
        date2=date(2012,1,9) # a MON close to next year begin
        assert cal.networkdays(date1,date2) == 17, 'networkdays'
        cal.addholidays(date(1,12,25)) #Christmas was a SUN in 2011
        assert cal.networkdays(date1,date2) == 17, 'networkdays'
        cal.addholidays(date(1,1,1)) #NewYear was also a SUN in 2012...
        assert cal.networkdays(date1,date2) == 17, 'networkdays'
        cal.addholidays(date(2012,1,2)) #...so the 2th was holiday
        assert cal.networkdays(date1,date2) == 16, 'networkdays'
        cal.addholidays(daysrange(date(2011,12,19),21)) #in fact, the company was closed 3 weeks
        assert cal.networkdays(date1,date2) == 2, 'networkdays'
        
                
        print cal.cast(datetime(2011,12,16,19))
        print cal.cast(datetime(2012,1,9,3))
        print cal.cast(datetime(2012,1,9,17))
        
        assert cal.diff(date1,date2)==timedelta(hours=8),'diff'
        assert cal.diff(date2,date1)==timedelta(hours=-8),'diff'
        assert cal.gethours(date1,date2)==8.
        
        assert cal.workday(date1,1) == date2, 'workday'
        assert cal.workday(date2,-1) == date1, 'workday %s != %s'%(cal.workday(date2,-1),date1)

