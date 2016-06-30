# coding: utf8

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

from Goulib.table import *
import datetime,os
import six

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
        assert_equal(self.t[0][0],datetime.date(2012, 6, 1))
        assert_equal(self.t[1][0],datetime.date(2012, 1,23))
        
        ref='<tr><td style="text-align:right;">2012-01-09</td><td>Central</td><td>Smith</td><td>Desk</td><td style="text-align:right;">2</td><td style="text-align:right;">125.00</td><td style="text-align:right;">250.00</td></tr>'
        t=Row(self.t[14]).html()
        assert_equal(t,ref)
        
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
    
    def test_read_csv(self):
        #test that t can be written to csv, then re-read in t2 without loss
        self.t.save(self.path+'/results/table.test.csv')
        
        t=Table(self.path+'/results/table.test.csv')
        t.to_date('OrderDate')

        assert_equal(t,self.t)
        
    def test_write_csv(self):
        pass #tested in test_read_csv
    
    def test_write_json(self):

        self.t.save(self.path+'/results/table.test.json',indent=True)
        t=Table(titles=self.t.titles) #to keep column order
        t.load(self.path+'/results/table.test.json')
        t.to_date('OrderDate')
        assert_equal(t,self.t)

    def test_read_xls(self):
        pass #tested in setup

    def test_to_date(self):
        pass #tested in setup and test_html

    def test_to_datetime(self):
        pass #tested in setup

    def test_html(self):
        t=self.t.save(self.path+'/results/test.htm')
        
        t=Table(self.path+'/results/test.htm')
        assert_equal(t._i('OrderDate'),0) #check the column exists
        
        t.to_date('OrderDate')
        assert_equal(t,self.t)
        
        #a table with objects in cells
        from Goulib.stats import Normal
        t=Table(titles=['A','B'],data=[[Normal(),2],[3,'last']])
        h=t.html()
        assert_true(h)
        
    def test_append(self):
        ta = Table()
        ta.append({'col1':1,'col2':2})
        assert_true(len(ta)==1 and ta.ncols()==2)
        ta.append([3,4])
        assert_true(len(ta)==2 and ta.ncols()==2)
        
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
        t=Table(self.t)
        n=len(t)
        t.addcol('Discount', 0.15, 4)
        assert_equal(len(t),n)  #check we don't change the lines
        assert_equal(t.ncols(),8)  

    def test_find_col(self):
        assert_equal(self.t.find_col('Date'),self.t._i('OrderDate'))

    def test_get(self):
        assert_equal(self.t.get(3,'Cost'),self.t[3][5])
        assert_equal(self.t.get(-1,'Total'),139.72)

    def test_groupby(self):
        d=self.t.groupby(u'Région')
        assert_equal(sum([len(d[k]) for k in d]),len(self.t))
        assert_equal(len(d['East']),13)

    def test___eq__(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.__eq__(other))
        raise SkipTest 

    def test_hierarchy(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.hierarchy(by, factory, linkfct))
        raise SkipTest 

    def test_icol(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.icol(by))
        raise SkipTest 

    def test_read_element(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_element(element, **kwargs))
        raise SkipTest 

    def test_read_html(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_html(filename, **kwargs))
        raise SkipTest 

    def test_remove_lines_where(self):
        import copy
        t=copy.deepcopy(self.t) #so we can play with it
        i=t._i('Rep')
        l=len(t)
        r=t.remove_lines_where(lambda line:line[i]=='Jones')
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
                        u'Unités': 75}
                     )

    def test_set(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.set(row, col, value))
        raise SkipTest 

    def test_total(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.total(funcs))
        raise SkipTest 

    def test_index(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.index(value, column))
        raise SkipTest
    
    def test_transpose(self):
        t=self.t.transpose()
        assert_equal(t[1,1],'Smith')


if __name__=="__main__":
    runmodule()
