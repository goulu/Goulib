#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.units import *

class Tests:
    def test_append_col(self):
        # assert_equal(expected, appendCol(self, colname, values))
        raise SkipTest # TODO: implement your test here

    def test001_simple_value(self):
        ureg.define('CHF = [Currency]')
        ureg.define('EUR = 1.21*CHF')
        ureg.define('USD = 0.93*CHF')
        dist = V(1000,'m')
        assert_equal(str(dist),'1000 meter')
        speed = V(10,'m/s')
        assert_equal(str(speed),'10 meter / second')
        time = dist/speed
        assert_equal(str(time),'100.0 second')
        hourlyRate = V(50,'USD/hour')
        cost = hourlyRate*time
        assert_equal(str(cost.to('CHF')),'1.29166666667 CHF')

    def test002_table(self):
        t = Table(   'mytable',      ['car',          'bus',                                     'pedestrian'],
                  [  'speed',        V(120,'km/hour'), V(100,'km/hour'),                          V(5,'km/hour'),
                     'acceleration', V(1,'m/s^2'),     V(0.1,'m/s^2'),                            V(0.2,'m/s^2'),
                     'autonomy',     V(600,'km'),      lambda: t['autonomy']['pedestrian']*10,    lambda: t['speed']['pedestrian']*V(6,'hour') #coucou
                  ])
        assert_count_equal(t.cols,['car',        'bus',         'pedestrian'])
        assert_count_equal(t.rowLabels,['speed','acceleration','autonomy'])
        from pprint import pformat#otherwise it takes Goulib.tests.pprint...
        logging.debug(pformat(t.rows))
        assert_equal(t['speed']['bus'],V(100,'km/hour'))
        logging.debug(t._repr_html_())
        
        v = View(t,rows=['autonomy','speed'],cols=['car','pedestrian'],rowUnits={'speed':'mile/hour'},name='my view')
        logging.debug(v._repr_html_())
        
        t.appendCol('cheval',{'speed':V(60,'km/hour'),'acceleration':V(0.3,'m/s^2'),'autonomy':V(40,'km')})
        logging.debug(t._repr_html_())
        
    def test003_m(self):
        v = V(60,'m/min')
        assert_equal(v('m/s'), 1)
        
if __name__ == "__main__":
    runmodule()