"""
WorkCalendar class with datetime operations on working hours, handling holidays
merges and improves `BusinessHours <http://pypi.python.org/pypi/BusinessHours/>`_ and `workdays <http://pypi.python.org/pypi/workdays/>`_ packages
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012, Philippe Guglielmetti"
__credits__ = ["http://pypi.python.org/pypi/BusinessHours",
               "http://pypi.python.org/pypi/workdays/",
               ]
__license__ = "LGPL"

import collections
import logging

from datetime import time, date, datetime, timedelta
from Goulib.datetime2 import timef, datef, datetimef, oneday, timedelta0, timedelta_div, timedelta_mul
from Goulib.interval import in_interval, intersection, intersectlen


class WorkCalendar(object):
    """WorkCalendar class with datetime operations on working hours"""

    # Define the weekday mnemonics to match the date.weekday function
    (MON, TUE, WED, THU, FRI, SAT, SUN) = list(range(7))

    def __init__(self, worktime=[time.min, time.max], parent=[], weekends=(SAT, SUN), holidays=set()):
        self.weekends = weekends
        self.holidays = set(holidays)
        if isinstance(parent, collections.Iterable):
            self.parents = parent
        else:
            self.parents = [parent]
        self.setworktime(worktime)
    start = property(fget=lambda self: self._worktime[0])
    end = property(fget=lambda self: self._worktime[1])

    def setworktime(self, worktime):
        self._worktime = list(map(timef, worktime))
        for p in self.parents:
            self._worktime = intersection(self._worktime, p._worktime)
        self.delta = timedelta(
            minutes=(self.end.hour-self.start.hour)*60+(self.end.minute-self.start.minute))
        if self.start == time.min and self.end == time.max:  # we have a microsecond delay
            self.delta = timedelta(hours=24)  # make it perfect

    def addholidays(self, days):
        """add day(s) to to known holidays. dates with year==4 (to allow Feb 29th) apply every year
        note : holidays set may contain weekends too."""
        try:  # iterable
            for day in days:
                self.holidays.add(day)
        except:
            self.holidays.add(days)
        return self

    def isworkday(self, day):
        """@return True if day is a work day"""
        if day.weekday() in self.weekends:
            return False
        if date(year=4, month=day.month, day=day.day) in self.holidays:
            return False
        if datef(day) in self.holidays:
            return False
        for p in self.parents:
            if not p.isworkday(day):
                return False
        return True

    def isworktime(self, time):
        """@return True if you're supposed to work at that time"""
        if not self.isworkday(time):
            return False
        return in_interval(self.workdatetime(time), time)

    def nextworkday(self, day):
        """@return next work day"""
        res = day
        while True:
            res = res+oneday
            if self.isworkday(res):
                break
        return res

    def prevworkday(self, day):
        """@return previous work day"""
        res = day
        while True:
            res = res-oneday
            if self.isworkday(res):
                break
        return res

    def range(self, start, end):
        """range of workdays between start (included) and end (not included)"""
        if start > end:
            return self.range(end, start)
        res = []
        day = start
        if not self.isworkday(day):
            day = self.nextworkday(start)
        while day < end:
            res.append(day)
            day = self.nextworkday(day)
        return res

    def workdays(self, start_date, ndays):
        """list of ndays workdays from start"""
        day = start_date
        res = [day]
        while ndays > 0:
            day = self.nextworkday(day)
            ndays = ndays-1
            res.append(day)
        while ndays < 0:
            day = self.prevworkday(day)
            ndays = ndays+1
            res.insert(0, day)
        return res

    def workday(self, start_date, ndays):
        '''Same as Excel WORKDAY function.
        Returns a date that is the indicated number of working days before or after the starting date. 
        Working days exclude weekends and any dates identified as holidays.
        Use WORKDAY to exclude weekends or holidays when you calculate invoice due dates, 
        expected delivery times, or the number of days of work performed.
        '''
        if ndays > 0:
            return self.workdays(start_date, ndays)[-1]
        else:
            return self.workdays(start_date, ndays)[0]

    def cast(self, time, retro=False):
        '''force time to be in workhours'''
        if self.isworktime(time):
            return time  # ok
        if retro:
            if not self.isworkday(time) or time.time() < self.start:
                return datetimef(self.prevworkday(time.date()), self.end)
            # only remaining case is time>self.end on a work day
            return datetimef(time.date(), self.end)
        else:
            if not self.isworkday(time) or time.time() > self.end:
                return datetimef(self.nextworkday(time.date()), self.start)
            # only remaining case is time<self.start on a work day
            return datetimef(time.date(), self.start)

    def worktime(self, day):
        '''@return interval of time worked a given day'''
        if not self.isworkday(day):
            return None
        return (self.start, self.end)

    def workdatetime(self, day):
        '''@return interval of datetime worked a given day'''
        if not self.isworkday(day):
            return None
        day = datef(day)
        return (datetimef(day, self.start), datetimef(day, self.end))

    def diff(self, t1, t2):
        '''@return timedelta worktime between t1 and t2 (= t2-t1)'''
        t1 = datetimef(t1, self.start)
        t2 = datetimef(t2, self.start)
        if t1 > t2:
            return -self.diff(t2, t1)
        fulldays = max(0, self.networkdays(t1, t2)-2)
        res = timedelta_mul(fulldays, self.delta)
        w1 = self.workdatetime(t1)
        if w1:
            res += intersectlen(w1, (t1, t2), timedelta0)
        w2 = self.workdatetime(t2)
        if w2:
            res += intersectlen(w2, (t1, t2), timedelta0)
        return res

    def gethours(self, t1, t2):
        '''@return fractional work hours between t1 and t2 (= t2-t1)'''
        return self.diff(t1, t2).total_seconds()/3600.

    def plus(self, start, t):
        '''@return start time + t work time (positive or negative)'''
        start = datetimef(start, self.start)
        if not self.isworktime(start):
            logging.error('%s is not in worktime' % start)
            raise ValueError
        days = timedelta_div(t, self.delta)
        res = start
        while days >= 1:
            res = self.nextworkday(res)
            days = days-1
        while days <= -1:
            res = self.prevworkday(res)
            days = days+1

        # less than one day of work
        remaining = timedelta_mul(self.delta, days)
        day = res.date()
        start = datetimef(day, self.start)
        end = datetimef(day, self.end)
        if (res+remaining) < start:  # skip to previous day
            remaining = (res+remaining)-start  # in full time
            res = datetimef(self.prevworkday(day), self.end)
        if (res+remaining) > end:  # skip to next day
            remaining = (res+remaining)-end  # in full time
            res = datetimef(self.nextworkday(day), self.start)
        return res+remaining

    def minus(self, start, t):
        '''@return start time - t work time (positive or negative)'''
        return self.plus(start, -t)

    def networkdays(self, start_date, end_date):
        '''Same as Excel NETWORKDAYS function.
        Returns the number of whole working days between 
        start_date and end_date (inclusive of both start_date and end_date). 
        Working days exclude weekends and any dates identified in holidays. 
        Use NETWORKDAYS to calculate employee benefits that accrue 
        based on the number of days worked during a specific term'''
        end_date = datef(end_date)
        start_date = datef(start_date)
        if end_date < start_date:
            return -self.networkdays(end_date, start_date)
        i = start_date
        res = 0
        while i <= end_date:
            if self.isworkday(i):
                res += 1
            i = datef(i+oneday)
        return res


''' a 24/24 7/7 calendar is useful'''
FullTime = WorkCalendar([time.min, time.max], holidays=[], weekends=[])

''' compatibility with http://pypi.python.org/pypi/BusinessHours'''


def workday(start_date, ndays, holidays=[]):
    '''Same as Excel WORKDAY function.
    Returns a date that is the indicated number of working days before or after the starting date. 
    Working days exclude weekends and any dates identified as holidays.
    Use WORKDAY to exclude weekends or holidays when you calculate invoice due dates, 
    expected delivery times, or the number of days of work performed.
    '''
    return WorkCalendar([8, 16], holidays).workday(start_date, ndays)


def networkdays(start_date, end_date, holidays=[]):
    '''Same as Excel NETWORKDAYS function.
    Returns the number of whole working days between 
    start_date and end_date (inclusive of both start_date and end_date). 
    Working days exclude weekends and any dates identified in holidays. 
    Use NETWORKDAYS to calculate employee benefits that accrue 
    based on the number of days worked during a specific term'''
    return WorkCalendar([8, 16], holidays).networkdays(start_date, end_date)
