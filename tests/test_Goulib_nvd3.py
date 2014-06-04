from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *

class TestGenerate:
    def test_generate(self):
        # assert_equal(expected, generate(y, x, length))
        raise SkipTest 

class TestChart:
    def test___init__(self):
        # chart = Chart(model, name, **kwargs)
        raise SkipTest 

    def test___str__(self):
        # chart = Chart(model, name, **kwargs)
        # assert_equal(expected, chart.__str__())
        raise SkipTest 

    def test_add(self):
        # chart = Chart(model, name, **kwargs)
        # assert_equal(expected, chart.add(y, name, x, **kwargs))
        raise SkipTest 

    def test_axis(self):
        # chart = Chart(model, name, **kwargs)
        # assert_equal(expected, chart.axis(name, label, format))
        raise SkipTest 

class TestLineChart:
    def test___init__(self):
        # line_chart = LineChart(**kwargs)
        raise SkipTest 

class TestScatterChart:
    def test___init__(self):
        # scatter_chart = ScatterChart(**kwargs)
        raise SkipTest 

class TestLineWithFocusChart:
    def test___init__(self):
        # line_with_focus_chart = LineWithFocusChart(**kwargs)
        raise SkipTest 

class TestMultiBarChart:
    def test___init__(self):
        # multi_bar_chart = MultiBarChart(**kwargs)
        raise SkipTest 

class TestMultiBarHorizontalChart:
    def test___init__(self):
        # multi_bar_horizontal_chart = MultiBarHorizontalChart(**kwargs)
        raise SkipTest 

class TestCumulativeLineChart:
    def test___init__(self):
        # cumulative_line_chart = CumulativeLineChart(**kwargs)
        raise SkipTest 

class TestStackedAreaChart:
    def test___init__(self):
        # stacked_area_chart = StackedAreaChart(**kwargs)
        raise SkipTest 

class TestLinePlusBarChart:
    def test___init__(self):
        # line_plus_bar_chart = LinePlusBarChart(**kwargs)
        raise SkipTest 

class TestMultiChart:
    def test___init__(self):
        # multi_chart = MultiChart(**kwargs)
        raise SkipTest 

class TestPareto:
    def test___init__(self):
        # pareto = Pareto(values, norm, **kwargs)
        raise SkipTest 

class TestHist:
    def test_hist(self):
        # assert_equal(expected, hist(values, bins))
        raise SkipTest 

class TestHistogram:
    def test___init__(self):
        # histogram = Histogram(values, **kwargs)
        raise SkipTest 

class TestBump:
    def test_bump(self):
        # assert_equal(expected, bump(n, w))
        raise SkipTest 

if __name__=="__main__":
    runmodule()