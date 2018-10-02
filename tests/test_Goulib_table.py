#!/usr/bin/env python
# coding: utf8

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them

from Goulib.tests import *
from Goulib.table import *
import datetime,os, operator,six

class TestTable:

    @classmethod
    def setup_class(self):
        self.path=os.path.dirname(os.path.abspath(__file__))

        #test reading an Excel file
        self.t=Table(self.path+'/data/test.xls') # from http://www.contextures.com/xlSampleData01.html
        assert_equal(self.t.titles,['OrderDate', u'Région', 'Rep', 'Item', u'Unités', 'Cost', 'Total'])

        #format some columns
        self.t.applyf('Cost',float)
        self.t.applyf('Total',lambda x:float(x) if isinstance(x,(six.integer_types,float)) else float(x.replace(',','')))
        self.t.to_date('OrderDate',fmt=['%m/%d/%Y','Excel']) #converts using fmts in sequence
        assert_equal(self.t[0,0],datetime.date(2012, 6, 1))
        assert_equal(self.t[1][0],datetime.date(2012, 1,23))

        #add a column to test timedeltas
        self.t.addcol('timedelta',self.t['OrderDate'] - self.t['OrderDate'].shift(-1))

    def test___init__(self):
        #most tests are above, but some more are here:
        #lists or tuples can be used
        t1=Table(([1,2],(3,4)))
        t2=Table([(1,2),[3,4]])
        assert_equal(t1,t2)
        #check Table is mutable even if instantiated with tuples
        t2[0][0]=2
        assert_not_equal(t1,t2)
        #and also generators, even for a single column
        assert_equal(Table((i for i in range(10))),Table((range(10))))

    def test___repr__(self):
        pass #tested in setup

    def test___str__(self):
        pass #tested in setup

    def test_applyf(self):
        pass #tested in setup

    def test_csv(self):
        #test that t can be written to csv, then re-read in t2 without loss
        self.t.save(self.path+'/results/table/test.csv')

        t=Table(self.path+'/results/table/test.csv')
        t.to_date('OrderDate')
        t.to_timedelta('timedelta')
        assert_equal(t,self.t)

    def test_json(self):
        p=self.path+'/results/table/test.json'
        self.t.save(p)
        #t=Table(titles=self.t.titles) #to keep column order
        t=Table(p)
        t.to_date('OrderDate')
        t.to_timedelta('timedelta')
        assert_equal(t,self.t)

    def test_xlsx(self):
        self.t.save(self.path+'/results/table/test.xlsx')

    def test_to_date(self):
        pass #tested in setup and test_html

    def test_to_datetime(self):
        pass #tested in setup

    def test_html(self):
        p=self.path+'/results/table/test.htm'

        self.t.save(p)
        t=Table(p)

        # t.to_date('OrderDate') #still required ?
        assert_equal(t,self.t)

        #a table with objects in cells
        from Goulib.stats import Normal
        t=Table(titles=['A','B'],data=[[Normal(),2],[3,'last']])
        h=t.html()
        assert_true(h)

    def test_append(self):
        ta = Table()
        ta.append({'col1':1,'col2':2})
        assert_equal(ta.shape,(1,2))
        ta.append([3,4])
        assert_equal(ta.shape,(2,2))

    def test_col(self):
        c1=self.t.col('Cost')
        c2=self.t['Cost']
        assert_equal(c1,c2)

    def test_sort(self):
        self.t.sort('Cost')
        col=self.t.col('Cost')
        assert_equal(col[0],1.29)
        assert_equal(col[-1],275)

    def test_ncols(self):
        assert_equal(self.t.ncols(),8)

    def test_addcol(self):
        t=Table(self.t)
        r,c=t.shape
        t.addcol('Discount', 0.15)
        assert_equal(t.shape,(r,c+1))

    def test_groupby(self):
        d=self.t.groupby(u'Région')
        assert_equal(sum([len(d[k]) for k in d]),len(self.t))
        assert_equal(len(d['East']),13)

    def test_remove_lines_where(self):
        t=Table(self.t) #creates a copy so we can play with it
        l=len(t)
        r=t.remove_lines_where('Rep','Jones')
        assert_equal(r,8)
        assert_equal(len(t),l-r)

    def test_rowasdict(self):
        r=self.t.rowasdict(3)
        assert_equal(r,{u'Cost': 1.99,
                        u'Item': u'Pencil',
                        u'OrderDate': date(2012,4,18),
                        u'Rep': u'Andrews',
                        u'Région': u'Central',
                        u'Total': 149.25,
                        u'Unités': 75,
                        u'timedelta':timedelta(days=105)}
                     )


if __name__=="__main__":
    runmodule()
