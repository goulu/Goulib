from goulib.tests import *
from goulib.table import *
import datetime
import os
import operator


class TestTable:

    @classmethod
    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(__file__))

        # test reading an Excel file
        # from http://www.contextures.com/xlSampleData01.html
        self.table = Table(self.path+'/data/test.xls')
        assert self.table.titles == ['OrderDate', 'Région', 'Rep', 'Item', 'Unités', 'Cost', 'Total']

        # format some columns
        self.table.applyf('Cost', float)
        self.table.applyf('Total', lambda x: float(x) if isinstance(
            x, (int, float)) else float(x.replace(',', '')))

        # converts using fmts in sequence
        self.table.to_date('OrderDate', fmt=['%m/%d/%Y', 'Excel'])
        assert self.table[0][0] == datetime.date(2012, 6, 1)
        assert self.table[1][0] == datetime.date(2012, 1, 23)

        # add a column to test timedeltas
        d = self.table.col('OrderDate')
        d = list(map(operator.sub, d, [d[0]]+d[:-1]))
        self.table.addcol('timedelta', d)

    def setup_method(self):
        # reset table to initial state
        self.t=Table(self.table)

    def test___init__(self):
        # most tests are above, but some more are here:
        # lists or tuples can be used
        t1 = Table(([1, 2], (3, 4)))
        t2 = Table([(1, 2), [3, 4]])
        assert t1 == t2
        # check Table is mutable even if instantiated with tuples
        t2[0][0] = 2
        assert t1 != t2
        # and also generators, even for a single column
        assert Table((i for i in range(10))) == Table((range(10)))

    def test___repr__(self):
        pytest.skip("not yet implemented")  # TODO: implement

    def test___str__(self):
        pytest.skip("not yet implemented")  # TODO: implement
    def test_applyf(self):
        pytest.skip("not yet implemented")  # TODO: implement

    def test_read_csv(self):
        # test that t can be written to csv, then re-read in t2 without loss
        self.t.save(self.path+'/results/table/test.csv')

        t = Table(self.path+'/results/table/test.csv')
        t.to_date('OrderDate')
        t.to_timedelta('timedelta')
        assert t == self.t

    def test_write_csv(self):
        pass  # tested in test_read_csv

    def test_write_json(self):

        self.t.save(self.path+'/results/table/test.json', indent=True)
        t = Table(titles=self.t.titles)  # to keep column order
        t.load(self.path+'/results/table/test.json')
        t.to_date('OrderDate')
        t.to_timedelta('timedelta')
        assert t == self.t

    def test_read_xls(self):
        pytest.skip("tested in setup")

    def test_write_xlsx(self):
        self.t.save(self.path+'/results/table/test.xlsx')

    def test_to_date(self):
        pytest.skip("tested in setup and test_html")

    def test_to_datetime(self):
        pytest.skip("tested in setup")

    def test_html(self):

        t = self.t.save(self.path+'/results/table/test.htm')

        t = Table(self.path+'/results/table/test.htm')
        assert t._i('OrderDate') == 0  # check the column exists

        t.to_date('OrderDate')
        t.to_timedelta('timedelta')
        assert t == self.t

        # a table with objects in cells
        from goulib.stats import Normal
        t = Table(titles=['A', 'B'], data=[[Normal(), 2], [3, 'last']])
        h = t.html()
        assert h

        # table with LaTex should not convert r'\right' in '\r ight'
        # and should add extra $$ to avoid line wraps in cells
        cell = r'${\left(x\right)}$'
        t = Table([[cell]])
        h = t.html()
        import re
        assert re.match('<table.*><tr.*><td.*>' +re.escape('$'+cell+'$')+'</td></tr>\n</table>\n',h)

    def test_append(self):
        ta = Table()
        ta.append({'col1': 1, 'col2': 2})
        assert len(ta) == 1 and ta.ncols() == 2
        ta.append([3, 4])
        assert len(ta) == 2 and ta.ncols() == 2

    def test_col(self):
        assert self.t.col('Cost') == self.t[:, 'Cost']

    def test_sort(self):
        self.t.sort('Cost')
        col = self.t.col('Cost')
        assert col[0] == 1.29
        assert col[-1] == 275

    def test_ncols(self):
        assert self.t.ncols() == 8

    def test_setcol(self):
        pass  # tested by test_addcol

    def test_addcol(self):
        t = Table(self.t)
        n = len(t)
        t.addcol('Discount', 0.15, 4)
        assert len(t) == n  # check we don't change the lines
        assert t.ncols() == 9

    def test_find_col(self):
        assert self.t.find_col('Date') == self.t._i('OrderDate')

    def test_get(self):
        assert self.t.get(3, 'Cost') == self.t[3, 5]
        assert self.t.get(-1, 'Total') == 139.72

    def test_groupby(self):
        d = self.t.groupby('Région')
        assert sum([len(d[k]) for k in d]) == len(self.t)
        assert len(d['East']) == 13

    def test___eq__(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.__eq__(other))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_hierarchy(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.hierarchy(by, factory, linkfct))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_icol(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.icol(by))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_read_element(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_element(element, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_read_html(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.read_html(filename, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_remove_lines_where(self):
        import copy
        t = copy.deepcopy(self.t)  # so we can play with it
        i = t._i('Rep')
        l = len(t)
        r = t.remove_lines_where(lambda line: line[i] == 'Jones')
        assert r == 8
        assert len(t) == l-r

    def test_rowasdict(self):
        r = self.t.rowasdict(6)
        assert r == {'Cost': 1.99,
                     'Item': 'Pencil',
                     'OrderDate': date(2012, 4, 18),
                     'Rep': 'Andrews',
                     'Région': 'Central',
                     'Total': 149.25,
                     'Unités': 75,
                     'timedelta': timedelta(days=105)}

    def test_set(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.set(row, col, value))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_total(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.total(funcs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_index(self):
        # table = Table(filename, titles, data, **kwargs)
        # assert_equal(expected, table.index(value, column))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_transpose(self):
        t = self.t.transpose()
        assert t[1, 1] == 'Jones'

    def test___getitem__(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.__getitem__(n))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_asdict(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.asdict())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_cols(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.cols(title))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_groupby_gen(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.groupby_gen(by, sort, removecol))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_json(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.json(**kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_load(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.load(filename, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_read_json(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.read_json(filename, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_save(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.save(filename, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_to_time(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.to_time(by, fmt, skiperrors))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_to_timedelta(self):
        # table = Table(data, **kwargs)
        # assert_equal(expected, table.to_timedelta(by, fmt, skiperrors))
        pytest.skip("not yet implemented")  # TODO: implement


class TestAttr:
    def test_attr(self):
        # assert_equal(expected, attr(args))
        pytest.skip("not yet implemented")  # TODO: implement


class TestCell:
    def test___init__(self):
        # cell = Cell(data, align, fmt, tag, style)
        pytest.skip("not yet implemented")  # TODO: implement

    def test___repr__(self):
        # cell = Cell(data, align, fmt, tag, style)
        # assert_equal(expected, cell.__repr__())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_html(self):
        # cell = Cell(data, align, fmt, tag, style)
        # assert_equal(expected, cell.html(**kwargs))
        pytest.skip("not yet implemented")  # TODO: implement

    def test_read(self):
        # cell = Cell(data, align, fmt, tag, style)
        # assert_equal(expected, cell.read())
        pytest.skip("not yet implemented")  # TODO: implement


class TestRow:
    def test___init__(self):
        # row = Row(data, align, fmt, tag, style)
        pytest.skip("not yet implemented")  # TODO: implement

    def test___repr__(self):
        # row = Row(data, align, fmt, tag, style)
        # assert_equal(expected, row.__repr__())
        pytest.skip("not yet implemented")  # TODO: implement

    def test_html(self):
        # row = Row(data, align, fmt, tag, style)
        # assert_equal(expected, row.html(cell_args, **kwargs))
        pytest.skip("not yet implemented")  # TODO: implement
