#! /usr/local/bin/python
# -*- coding: utf-8 -*-

import os
from nose import SkipTest
from nose.tools import assert_equal, assert_true

from Goulib.table import *
import datetime

class TestTable:
    
    @classmethod
    def setup_class(self):
        self.path=os.path.dirname(os.path.abspath(__file__))
        
        #test reading an Excel file
        self.t=Table(self.path+'/test.xls') # from http://www.contextures.com/xlSampleData01.html
        assert_equal(self.t.titles,['OrderDate', 'Region', 'Rep', 'Item', 'Units', 'Cost', 'Total'])
        
        #test that t can be written to csv, then re-read in t2 without loss
        self.t.write_csv(self.path+'/test.csv')
        
        self.t2=Table(None) #empty table
        self.t2.read_csv(self.path+'/test.csv')
        
        assert_equal(repr(self.t),repr(self.t2))
        assert_equal(str(self.t),str(self.t2))
        
        #format some columns
        self.t2.applyf('Cost',float)
        self.t2.applyf('Total',lambda x:float(x) if isinstance(x,(int,float)) else float(x.replace(',','')))
        
        self.t2.to_date('OrderDate',fmt='%m/%d/%Y',skiperrors=True) #converts string format
        self.t2.to_date('OrderDate',fmt='') #converts Excel numeric format
        assert_equal(self.t2[0][0],datetime.date(2012, 6, 1))
        assert_equal(self.t2[1][0],datetime.date(2012, 1,23))
        
        ref='<tr><td align="right">2012-01-09</td><td>Central</td><td>Smith</td><td>Desk</td><td align="right">2</td><td align="right">125.00</td><td align="right">250.00</td></tr>'
        assert_equal(Row(self.t2[14]).html(),ref)
        
    def test___init__(self):
        pass #tested in setup

    def test___repr__(self):
        pass #tested in setup

    def test___str__(self):
        pass #tested in setup

    def test_applyf(self):
        pass #tested in setup
    
    def test_read_csv(self):
        pass #tested in setup

    def test_read_xls(self):
        pass #tested in setup

    def test_to_date(self):
        pass #tested in setup and test_html

    def test_to_datetime(self):
        pass #tested in setup

    def test_write_csv(self):
        pass #tested in setup

    def test_html(self):
        f = open(self.path+'/test.htm', 'w')
        f.write(self.t2.html())
        f.close()
        
        t=Table(self.path+'/test.htm')
        assert_equal(t._i('OrderDate'),0) #check the colum exists
        
        t.to_date('OrderDate')
        assert_equal(t,self.t2)
        
    def test_append(self):
        ta = Table(None)
        ta.append({'col1':1,'col2':2})
        assert_true(len(ta)==1 and ta.ncols()==2)
        ta.append([3,4])
        assert_true(len(ta)==2 and ta.ncols()==2)
        
    def test_col(self):
        pass #tested by test_sort
        
    def test_sort(self):
        self.t2.sort('Cost')
        col=self.t2.col('Cost')
        assert_equal(col[0],1.29)
        assert_equal(col[-1],275)
        
    def test_ncols(self):
        assert_equal(self.t.ncols(),7)
        
    def test_setcol(self):
        pass #tested by test_addcol

    def test_addcol(self):
        n=len(self.t2)
        self.t2.addcol('Discount', 0.15, 4)
        assert_equal(len(self.t2),n)  #check we don't change the lines
        assert_equal(self.t2.ncols(),8)  

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

    def test_read_element(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_element(element, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_read_html(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_html(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

class TestHtml:
    def test_html(self):
        # assert_equal(expected, html(self))
        raise SkipTest # TODO: implement your test here

class TestAttr:
    def test_attr(self):
        # assert_equal(expected, attr(args))
        raise SkipTest # TODO: implement your test here

class TestRead:
    def test_read(self):
        # assert_equal(expected, read(x))
        raise SkipTest # TODO: implement your test here

if __name__=="__main__":
    import sys
    import nose
    from cStringIO import StringIO  
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0], module_name, '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()
