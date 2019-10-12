#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.workdays import *

class TestWorkCalendar:
    @classmethod
    def setup_class(self):
        self.base=WorkCalendar() 
        #derive a specific calendar, with "inherited" weekends for deeper tests
        self.cal=WorkCalendar([8,16],parent=self.base,weekends=[])
        self.cal.addholidays(date(1,12,25)) #Christmas was a SUN in 2011
        self.cal.addholidays(date(1,1,1)) #NewYear was also a SUN in 2012...
        self.cal.addholidays(date(2012,1,2)) #...so the 2th was holiday
        start=date(2011,12,18) #start of holidays
        end=date(2012,1,8) #the company was closed 3 weeks
        closed=self.base.range(end,start)
        self.cal.addholidays(closed) 

    def test___init__(self):
        pass #tested above

    def test_addholidays(self):
        date1=date(2011,12,16) # a FRI close to year end and holidays
        date2=date(2012,1,9) # a MON close to next year begin
        #calendar is inited with addholidays in setup_class above
        assert_equal(self.cal.networkdays(date1,date2),2)

    def test_cast(self):
        cal=WorkCalendar([8,16])
        assert_equal(cal.cast(datetime(2012,1,9,10)),datetime(2012,1,9,10))
        assert_equal(cal.cast(datetime(2012,1,9,3)),datetime(2012,1,9,8))
        assert_equal(cal.cast(datetime(2012,1,9,3),retro=True),datetime(2012,1,6,16)) #TODO: check if it's correct
        assert_equal(cal.cast(datetime(2012,1,9,17)),datetime(2012,1,10,8))
        #cast friday 19h to next monday 8h
        assert_equal(cal.cast(datetime(2011,12,16,19)),datetime(2011,12,19,8))
        assert_equal(cal.cast(datetime(2011,12,16,19),retro=True),datetime(2011,12,16,16))

    def test_diff(self):
        date1=date(2011,12,16) # a FRI close to year end and holidays
        date2=date(2012,1,9) # a MON close to next year begin
        assert_equal(self.cal.diff(date1,date2),timedelta(hours=8))
        assert_equal(self.cal.diff(date2,date1),timedelta(hours=-8))

    def test_gethours(self):
        date1=date(2011,12,16) # a FRI close to year end and holidays
        date2=date(2012,1,9) # a MON close to next year begin
        assert_equal(self.cal.gethours(date1,date2),8)

    def test_isworkday(self):
        assert_true(self.cal.isworkday(date(2011,12,16)))
        assert_false(self.cal.isworkday(date(2011,12,17)))

    def test_isworktime(self):
        assert_true(self.cal.isworktime(datetime(2012,1,9,8,0,0)))
        assert_false(self.cal.isworktime(datetime(2012,1,9,7,59,59)))

    def test_minus(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.minus(start, t))
        raise SkipTest 

    def test_networkdays(self):
        date1=date(2011,12,16) # a FRI close to year end and holidays
        assert_equal(self.cal.networkdays(date1,date1),1)
        assert_equal(self.cal.networkdays(date1,date(2011,12,17)),1)
        assert_equal(self.cal.networkdays(date1,date(2011,12,18)),1)
        assert_equal(self.cal.networkdays(date1,date(2011,12,19)),1)

    def test_nextworkday(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.nextworkday(day))
        raise SkipTest 

    def test_plus(self):
        start=date(2011,12,19) #start of holidays
        end=self.base.plus(start,timedelta(days=21)) #the company was closed 3

    def test_prevworkday(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.prevworkday(day))
        raise SkipTest 

    def test_range(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.range(start, end))
        raise SkipTest 

    def test_setworktime(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.setworktime(worktime))
        raise SkipTest 

    def test_workdatetime(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.workdatetime(day))
        raise SkipTest 

    def test_workday(self):
        date1=date(2011,12,16) # a FRI close to year end and holidays
        date2=date(2012,1,9) # a MON close to next year begin
        assert_equal(self.cal.networkdays(date1,date2),2)
        assert_equal(self.cal.workday(date1,1),date2)
        assert_equal(self.cal.workday(date2,-1),date1)

    def test_workdays(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.workdays(start_date, ndays))
        raise SkipTest 

    def test_worktime(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.worktime(day))
        raise SkipTest 

class TestWorkday:
    def test_workday(self):
        pass #tested above

class TestNetworkdays:
    def test_networkdays(self):
        # assert_equal(expected, networkdays(start_date, end_date, holidays))
        raise SkipTest 
    

if __name__=="__main__":
    runmodule()
