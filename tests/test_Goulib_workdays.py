from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.workdays import *

class TestWorkCalendar:
    def test___init__(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        raise SkipTest 

    def test_addholidays(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.addholidays(days))
        raise SkipTest 

    def test_cast(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.cast(time, retro))
        raise SkipTest 

    def test_diff(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.diff(t1, t2))
        raise SkipTest 

    def test_gethours(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.gethours(t1, t2))
        raise SkipTest 

    def test_isworkday(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.isworkday(day))
        raise SkipTest 

    def test_isworktime(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.isworktime(time))
        raise SkipTest 

    def test_minus(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.minus(start, t))
        raise SkipTest 

    def test_networkdays(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.networkdays(start_date, end_date))
        raise SkipTest 

    def test_nextworkday(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.nextworkday(day))
        raise SkipTest 

    def test_plus(self):
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.plus(start, t))
        raise SkipTest 

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
        # work_calendar = WorkCalendar(worktime, parent, weekends, holidays)
        # assert_equal(expected, work_calendar.workday(start_date, ndays))
        raise SkipTest 

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
        # assert_equal(expected, workday(start_date, ndays, holidays))
        raise SkipTest 

class TestNetworkdays:
    def test_networkdays(self):
        # assert_equal(expected, networkdays(start_date, end_date, holidays))
        raise SkipTest 

if __name__=="__main__":
    runmodule()
