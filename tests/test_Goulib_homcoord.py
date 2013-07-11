#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2013, Philippe Guglielmetti"
__license__ = "LGPL"

import nose
from nose.tools import *
from nose import SkipTest
from nose.tools import assert_equal

from Goulib.homcoord import *

def assert_almost_equal(a,b,precision=6): # allow tests on Pt with 6 decimals precision
    map(lambda x:nose.tools.assert_almost_equal(x[0],x[1],precision),zip(a,b))

class TestPt:
    def setup(self):
        self.pt0 = Pt(0,0)
        self.pt1 = Pt(3,4)
        self.pt2 = Pt((6.,8.0,2.)) # same as pt1, but with scale factor
        
    def test___cmp__(self):
        assert_equal(True, self.pt1==self.pt2)
        
    def test_x(self):
        assert_equal(3, self.pt2.x)

    def test_y(self):
        assert_equal(4, self.pt2.y)
        
    def test_xy(self):
        assert_equal((3,4), self.pt2.xy)

    def test___repr__(self):
        assert_equal("(3, 4)", repr(self.pt2))

    def test___str__(self):
        assert_equal("(3, 4)", str(self.pt2))
        
    def test___add__(self):
        assert_equal(Pt(6,8), self.pt1+self.pt2)
            
    def test___sub__(self):
        assert_equal(self.pt0, self.pt1-self.pt2)
        
    def test_dist(self):
        assert_equal(5, self.pt1.dist(self.pt0))

    def test_bearing(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.bearing(p))
        raise SkipTest # TODO: implement your test here

    def test_radial(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.radial(d, bearing))
        raise SkipTest # TODO: implement your test here

    def test_toPolar(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.toPolar())
        raise SkipTest # TODO: implement your test here
    
    def test___div__(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.__div__(scale))
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # pt = Pt(*args)
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.__mul__(scale))
        raise SkipTest # TODO: implement your test here

    def test_apply(self):
        # pt = Pt(*args)
        # assert_equal(expected, pt.apply(f))
        raise SkipTest # TODO: implement your test here

class TestXlate:
    def test_xlate(self):
        dx=2
        dy=3
        m1=Xform([(1,0,dx),(0,1,dy),(0,0,1)])
        m2=Xlate((dx,dy))
        assert_equal(m1._m, m2._m)

class TestXscale:
    def test_xscale(self):
        sx=2
        sy=3
        m1=Xform([(sx,0,0),(0,sy,0),(0,0,1)])
        m2=Xscale((sx,sy))
        assert_equal(m1._m, m2._m)

class TestXrotate:
    def test_xrotate(self):
        m=Xrotate(pi/4) # 45°
        v=m(Xform.UNIT)
        assert_almost_equal([0,1], v.xy)

class TestXrotaround:
    def test_xrotaround(self):
        # assert_equal(expected, Xrotaround(p, theta))
        raise SkipTest # TODO: implement your test here

class TestXform:
    
    def setup(self):
        self.m1=Xform([[1,0,0],[0,1,0],[0,0,1]]) #identity
        dx=2
        dy=3
        self.mt=Xlate((dx,dy))
        self.mr=Xrotate(pi/4) # 45°
        sx=2
        sy=3
        self.ms=Xscale((sx,sy))
        self.m=self.m1*self.mt*self.mr*self.ms
        
    def test_apply(self):
        assert_almost_equal(Pt(0,12.727922061357855).xy,self.m(Pt(1,0)).xy)
        assert_almost_equal(Pt(-2.8284271247461894,12.727922061357855).xy,self.m(Pt(0,1)).xy)
        
    def test___call__(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.__call__(p))
        raise SkipTest # TODO: implement your test here

    def test___mul__(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.__mul__(other))
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.__str__())
        raise SkipTest # TODO: implement your test here

    def test_angle(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.angle())
        raise SkipTest # TODO: implement your test here

    def test_compose(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.compose(t2))
        raise SkipTest # TODO: implement your test here

    def test_inverse(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.inverse())
        raise SkipTest # TODO: implement your test here

    def test_invert(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.invert(p))
        raise SkipTest # TODO: implement your test here

    def test_mag(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.mag())
        raise SkipTest # TODO: implement your test here

    def test_offset(self):
        # xform = Xform(m)
        # assert_equal(expected, xform.offset())
        raise SkipTest # TODO: implement your test here

    def test___init__(self):
        # xform = Xform(m)
        raise SkipTest # TODO: implement your test here

class TestPolar:
    def test___init__(self):
        # polar = Polar(*p)
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # polar = Polar(*p)
        # assert_equal(expected, polar.__str__())
        raise SkipTest # TODO: implement your test here

    def test_toCartesian(self):
        # polar = Polar(*p)
        # assert_equal(expected, polar.toCartesian())
        raise SkipTest # TODO: implement your test here

class TestLine:
    def test___init__(self):
        # line = Line(a, b, c)
        raise SkipTest # TODO: implement your test here

    def test___str__(self):
        # line = Line(a, b, c)
        # assert_equal(expected, line.__str__())
        raise SkipTest # TODO: implement your test here

    def test_intersect(self):
        # line = Line(a, b, c)
        # assert_equal(expected, line.intersect(other))
        raise SkipTest # TODO: implement your test here

    def test_pointBearing(self):
        # line = Line(a, b, c)
        # assert_equal(expected, line.pointBearing(bears))
        raise SkipTest # TODO: implement your test here

    def test_twoPoint(self):
        # line = Line(a, b, c)
        # assert_equal(expected, line.twoPoint(p2))
        raise SkipTest # TODO: implement your test here

class TestArgPair:
    def test_arg_pair(self):
        # assert_equal(expected, argPair(*p))
        raise SkipTest # TODO: implement your test here

class TestNormAngle:
    def test_norm_angle(self):
        # assert_equal(expected, normAngle(theta))
        raise SkipTest # TODO: implement your test here

if __name__ == "__main__":
    nose.runmodule()