#! /usr/local/bin/python
# -*- coding: utf-8 -*-

import os
from nose import SkipTest
from nose.tools import assert_equal

from Goulib.table import *

class TestTable:
    
    def setup(self):
        self.path=os.path.dirname(os.path.abspath(__file__))
        self.t=Table(self.path+'\\test.xls') # from http://www.contextures.com/xlSampleData01.html
        
    def test___init__(self):
        assert_equal(self.t.titles,['OrderDate', 'Region', 'Rep', 'Item', 'Units', 'Cost', 'Total'])
    
    def test_read_xls(self):
        pass #tested in setup
    
    def test_write_csv(self):
        self.t.write_csv(self.path+'\\test.csv')
        
    def test_read_csv(self):
        t2=Table(None) #empty table
        t2.read_csv(self.path+'\\test.csv')
        assert_equal(self.t, t2)
        
    def test___repr__(self):
        t2=Table(self.path+'\\test.csv')
        assert_equal(repr(self.t), repr(t2))

    def test___str__(self):
        t2=Table(self.path+'\\test.csv')
        assert_equal(str(self.t), str(t2))
        
    def test_applyf(self):
        pass #tested by test_to_datetime
        
    def test_to_datetime(self):
        pass #tested by test_to_date
    
    def test_to_date(self):
        import datetime
        self.t.to_date('OrderDate',fmt='%m/%d/%Y') #converts string format
        self.t.to_date('OrderDate',fmt='',safe=False) #converts Excel numeric format
        assert_equal(self.t[0][0],datetime.date(2012, 6, 1))
        assert_equal(self.t[1][0],datetime.date(2012, 1,23))
        
    def test_col(self):
        pass #tested by test_sort
        
    def test_sort(self):
        self.t.sort('Cost')
        col=self.t.col('Cost')
        assert_equal(col[0],1.29)
        assert_equal(col[-1],275)
        
    def test_ncols(self):
        assert_equal(self.t.ncols(),7)
        
    def test_setcol(self):
        pass #tested by test_addcol

    def test_addcol(self):
        self.t.addcol('Discount', 0.15, 4)
        assert_equal(self.t.ncols(),8)    

    def test_find_col(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.find_col(title))
        raise SkipTest # TODO: implement your test here

    def test_get(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.get(row, col))
        raise SkipTest # TODO: implement your test here

    def test_groupby(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.groupby(by, sort, removecol))
        raise SkipTest # TODO: implement your test here

    def test_hierarchy(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.hierarchy(by, factory, linkfct))
        raise SkipTest # TODO: implement your test here

    def test_html(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.html(page, head, foot, colstyle, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_remove_lines_where(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.remove_lines_where(filter))
        raise SkipTest # TODO: implement your test here

    def test_rowasdict(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.rowasdict(i))
        raise SkipTest # TODO: implement your test here

    def test_set(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.set(row, col, value))
        raise SkipTest # TODO: implement your test here


    def test_total(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.total(funcs))
        raise SkipTest # TODO: implement your test here
    
    def test_icol(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.icol(by))
        raise SkipTest # TODO: implement your test here



    def test___eq__(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.__eq__(other))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
