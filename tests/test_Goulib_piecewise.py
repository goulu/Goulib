from nose.tools import assert_equal, assert_almost_equal
from nose import SkipTest
from Goulib.piecewise import *

class TestPiecewise:
    def test___add__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__add__(other))
        raise SkipTest # TODO: implement your test here

    def test___and__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__and__(other))
        raise SkipTest # TODO: implement your test here

    def test___call__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__call__(x))
        raise SkipTest # TODO: implement your test here

    def test___div__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__div__(other))
        raise SkipTest # TODO: implement your test here

    def test___getitem__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__getitem__(i))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # piecewise = Piecewise(init, default, start)
        raise SkipTest # TODO: implement your test here

    def test___iter__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__iter__())
        raise SkipTest # TODO: implement your test here

    def test___len__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__len__())
        raise SkipTest # TODO: implement your test here

    def test___lshift__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__lshift__(dx))
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___neg__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__neg__())
        raise SkipTest # TODO: implement your test here

    def test___not__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__not__())
        raise SkipTest # TODO: implement your test here

    def test___or__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__or__(other))
        raise SkipTest # TODO: implement your test here

    def test___rshift__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__rshift__(dx))
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__str__())
        raise SkipTest # TODO: implement your test here

    def test___sub__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__sub__(other))
        raise SkipTest # TODO: implement your test here

    def test___xor__(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.__xor__(other))
        raise SkipTest # TODO: implement your test here

    def test_append(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.append(item))
        raise SkipTest # TODO: implement your test here

    def test_applx(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.applx(f))
        raise SkipTest # TODO: implement your test here

    def test_apply(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.apply(f))
        raise SkipTest # TODO: implement your test here

    def test_extend(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.extend(iterable))
        raise SkipTest # TODO: implement your test here

    def test_index(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.index(x, v))
        raise SkipTest # TODO: implement your test here

    def test_lines(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.lines(min, max, eps))
        raise SkipTest # TODO: implement your test here

    def test_list(self):
        # piecewise = Piecewise(init, default, start)
        # assert_equal(expected, piecewise.list())
        raise SkipTest # TODO: implement your test here

"""
import unittest
class TestCase(unittest.TestCase):       
    def setUp(self):
        import Goulib.markup as markup
        
        self.page=markup.page()
        self.page.init(
                doctype="Content-Type: text/html; charset=utf-8\r\n\r\n<!DOCTYPE html>",
                script=["http://nvd3.org/lib/d3.v2.js","http://nvd3.org/nv.d3.js"], #must be a list to preserve order
                css=['http://nvd3.org/src/nv.d3.css']
                )
         
    def runTest(self):
        from nvd3 import LineChart
        from colors import color_range
        fig=LineChart(height=400,colors=color_range(6,'red','blue'))
        def add(p,name,min=0,max=10,disabled=False):
            print name,'=',p,'<br/>'
            x,y=p.lines(min=min,max=max)
            fig.add(x=x,y=y,name=name,disabled=disabled)
            
        p1=Piecewise([(4,4),(3,3),(1,1),(5,0)])
        self.assertEqual(str(p1),'[(-inf, 0), (1, 1), (3, 3), (4, 4), (5, 0)]')
        add(p1,'p1')
        p2=Piecewise(default=1)
        p2+=(2.5,1,6.5)
        self.assertEqual(str(p2),'[(-inf, 1), (2.5, 2), (6.5, 1)]')
        add(p2,'p2')
        add(p1+p2,'p1+p2',disabled=True)
        add(p1-p2,'p1-p2',disabled=True)
        add(p1*p2,'p1*p2',disabled=True)
        p1.apply(float) #to make division correct
        add(p1/p2,'p1/p2',disabled=True)
        self.page.add(str(fig))
        
        dx=0.05 #small shift to see curves better
        fig=LineChart(colors=fig.colors)
        b1=Piecewise([(2,True)],False)
        add(b1,'b1')
        b2=Piecewise([(1,True),(2,False),(3,True)],False)
        add(b2<<dx,'b2')
        add((b1 | b2)>>dx,'b1 or b2',disabled=True)
        add((b1 & b2)>>2*dx,'b1 and b2',disabled=True)
        add((b1 ^ b2)<<3*dx,'b1 xor b2',disabled=True)
        self.page.add(str(fig))

        return
        from datetime import datetime,timedelta
        self.ptime=Piecewise([(timedelta(hours=4),4),(timedelta(hours=3),3),(timedelta(hours=1),1),(timedelta(hours=2),2),(timedelta(hours=5),0)])

    def tearDown(self):
        print self.page
"""

if __name__ == "__main__":
    import nose
    nose.runmodule()
