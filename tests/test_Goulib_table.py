from nose import SkipTest
from nose.tools import assert_equal

class TestTable:
    def test___init__(self):
        # table = Table(filename, titles, init, **kwargs)
        raise SkipTest # TODO: implement your test here

    def test___repr__(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.__repr__())
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.__str__())
        raise SkipTest # TODO: implement your test here

    def test_addcol(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.addcol(title, val, i))
        raise SkipTest # TODO: implement your test here

    def test_applyf(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.applyf(by, f, safe))
        raise SkipTest # TODO: implement your test here

    def test_col(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.col(by))
        raise SkipTest # TODO: implement your test here

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

    def test_icol(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.icol(by))
        raise SkipTest # TODO: implement your test here

    def test_ncols(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.ncols())
        raise SkipTest # TODO: implement your test here

    def test_read_csv(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.read_csv(filename, **kwargs))
        raise SkipTest # TODO: implement your test here

    def test_read_xls(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.read_xls(filename, **kwargs))
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

    def test_setcol(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.setcol(by, val, i))
        raise SkipTest # TODO: implement your test here

    def test_sort(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.sort(by, reverse))
        raise SkipTest # TODO: implement your test here

    def test_to_date(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.to_date(by, fmt, safe))
        raise SkipTest # TODO: implement your test here

    def test_to_datetime(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.to_datetime(by, fmt, safe))
        raise SkipTest # TODO: implement your test here

    def test_total(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.total(funcs))
        raise SkipTest # TODO: implement your test here

    def test_write_csv(self):
        # table = Table(filename, titles, init, **kwargs)
        # assert_equal(expected, table.write_csv(filename, transpose, **kwargs))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    import nose
    nose.runmodule()
