#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.datetime2 import *

class TestDatef:
    def test_datef(self):
        d=date(year=1963,month=12,day=25)
        assert_equal(datef(d),d)
        assert_equal(datef('1963-12-25'),d)
        assert_equal(datef('25/12/1963',fmt='%d/%m/%Y'),d)
        assert_equal(datef(40179,fmt=None),date(year=2010,month=1,day=1)) # http://answers.oreilly.com/topic/1694-how-excel-stores-date-and-time-values/
        assert_equal(datef(40179,fmt='Excel'),date(year=2010,month=1,day=1)) # http://answers.oreilly.com/topic/1694-how-excel-stores-date-and-time-values/

class TestTimef:
    def test_timef(self):
        t=time(hour=12,minute=34,second=56)
        assert_equal(timef(t),t)
        assert_equal(timef('12:34:56'),t)
        assert_true(equal(timef(0.503473,fmt=None),time(hour=12,minute=0o5))) #http://answers.oreilly.com/topic/1694-how-excel-stores-date-and-time-values/
    
class TestDatetimef:
    def test_datetimef(self):
        d=datetime(year=1963,month=12,day=25,hour=12,minute=34,second=56)
        t=d.time()
        assert_equal(datetimef(d),d)
        assert_equal(datetimef('1963-12-25',t),d)
        assert_equal(datetimef('25/12/1963',t,fmt='%d/%m/%Y'),d)
        assert_true(equal(datetimef(40179.503472,fmt=None),datetime(year=2010,month=0o1,day=0o1,hour=12,minute=0o5))) #http://answers.oreilly.com/topic/1694-how-excel-stores-date-and-time-values/

class TestTimedeltaf:
    def test_timedeltaf(self):
        td=timedelta(days=25,hours=12,minutes=34,seconds=56)
        s=str(td)
        s=timedeltaf(s)
        assert_equal(s,td)
        #check with hours > 24
        s=('%d:%d:%d'%(25*24+12,34,56))
        s=timedeltaf(s)
        assert_equal(s,td)
        #check negative
        td=timedeltaf('-1 day,00:00:00')
        s=str(td)
        s=timedeltaf(s)
        assert_equal(s,td)
        

class TestStrftimedelta:
    def test_strftimedelta(self):
        d1=datetime(year=1963,month=12,day=25,hour=12,minute=34,second=56)
        d2=datetime(year=1965,month=6,day=13,hour=0o1,minute=23,second=45)
        td=d2-d1
        assert_equal(strftimedelta(td),'12852:48:49')

class TestTdround:
    def test_tdround(self):
        td=timedelta(seconds=10000)
        assert_equal(tdround(td, 60),timedelta(seconds=10020))
        
class TestMinutes:
    def test_minutes(self):
        td=timedelta(seconds=10000)
        assert_equal(minutes(td),10000./60)

class TestHours:
    def test_hours(self):
        td=timedelta(seconds=10000)
        assert_equal(hours(td),10000./3600)

class TestDaysgen:
    def test_daysgen(self):
        start=datetime(year=1963,month=12,day=25,hour=12,minute=34,second=56)
        for d in daysgen(start,10): pass
        assert_equal(d, datetime(year=1964,month=1,day=3,hour=12,minute=34,second=56))
        for d in daysgen(start,15,onehour): pass
        assert_equal(d, datetime(year=1963,month=12,day=26,hour=2,minute=34,second=56))

class TestDays:
    def test_days(self):
        r=days(datetime.today(),21)
        assert_equal(len(r), 21)

class TestTimedeltaSum:
    def test_timedelta_sum(self):
        td=timedelta(seconds=10000)
        assert_equal(timedelta_sum([td]*6),timedelta(seconds=60000))
        assert_equal(timedelta_sum([]),timedelta0)

class TestTimedeltaDiv:
    def test_timedelta_div(self):
        assert_equal(timedelta_div(oneday, onehour),24)

class TestTimedeltaMul:
    def test_timedelta_mul(self):
        assert_equal(timedelta_mul(onehour, 24),oneday)

class TestTimeSub:
    def test_time_sub(self):
        t1=time(hour=12,minute=34,second=56)
        t2=time(hour=0o1,minute=23,second=45)
        assert_equal(time_sub(t1, t2),timedelta(seconds=40271))

class TestTimeAdd:
    def test_time_add(self):
        t1=time(hour=12,minute=34,second=56)
        t2=time(hour=0o1,minute=23,second=45)
        assert_equal(time_add(t2, timedelta(seconds=40271)),t1)

class TestDatetimeIntersect:
    def test_datetime_intersect(self):
        start=datetime(year=1963,month=12,day=25,hour=12,minute=34,second=56)
        td=timedelta(seconds=40271)
        d=[d for d in daysgen(start,4,td)]
        assert_equal(datetime_intersect([d[0],d[2]],[d[1],d[3]]),td)

class TestTimeIntersect:
    def test_time_intersect(self):
        start=time(hour=12,minute=34,second=56)
        td=timedelta(seconds=1271)
        d=[d for d in daysgen(start,4,td)]
        assert_equal(time_intersect([d[0],d[2]],[d[1],d[3]]),td)


class TestDatetime:
    def test___sub__(self):
        # datetime = datetime()
        # assert_equal(expected, datetime.__sub__(other))
        raise SkipTest # TODO: implement your test here

class TestTimedelta:
    def test_isoformat(self):
        # timedelta = timedelta()
        # assert_equal(expected, timedelta.isoformat())
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    runmodule()
