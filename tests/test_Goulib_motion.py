from __future__ import division #"true division" everywhere

from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.motion import *

def pva_almost_equal(a,b,precision=6): # allow tests on Pt with 6 decimals precision
    map(lambda x:assert_almost_equal(x[0],x[1],precision),zip(a,b))
    
class TestPVA:
    def test___init__(self):
        pass # tested below
    
    def test___call__(self):
        pass # tested below
    
class TestSegment:
    def test___init__(self):
        pass # tested below
    
    def test___call__(self):
        pass # tested below

    def test_dt(self):
        pass # tested below

    def test_end(self):
        pass # tested below

    def test_start(self):
        pass # tested below
    
class TestSegmentPoly:
    @classmethod
    def setup_class(self):
        self.seg=SegmentPoly(1,2,[1,2,3])
        
    def test_start(self):
        pva=self.seg.start()
        assert_equal(pva,(1,2,6,0))
        
    def test_end(self):
        pva=self.seg.end()
        assert_equal(pva,(6,8,6,0))
    
    def test___call__(self):
        assert_equal(self.seg(.9),(0,0,0,0))
        assert_equal(self.seg(1),(1,2,6,0))
        pva_almost_equal(self.seg(2-1e-12),(6,8,6,0))
        assert_equal(self.seg(2),(0,0,0,0))
        
    def test___init__(self):
        # segment_poly = SegmentPoly(t0, t1, p)
        raise SkipTest 

class TestSegment2ndDegree:
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 1,2      
        self.p0, self.v0, self.a = -1,1,2
        self.start=(self.p0, self.v0, self.a, 0)
        self.p1, self.v1, self.a1 = 1,3,self.a
        self.end=(self.p1, self.v1, self.a1, 0)
        
    def test1(self):
        t0,t1,p0,v0,p1 = self.t0,self.t1,self.p0,self.v0,self.p1
        seg=Segment2ndDegree(t0,t1,(p0,v0),p1) # time interval and start + end positions  + initial velocity
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def test2(self):
        t0,t1,p0,v0,a = self.t0,self.t1,self.p0,self.v0,self.a
        seg=Segment2ndDegree(t0,t1,(p0,v0,a)) # time interval and start pva
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
    
    def test3(self):
        t0,t1,p1,v1,a = self.t0,self.t1,self.p1,self.v1,self.a
        seg=Segment2ndDegree(t0,t1,None,(p1,v1,a)) # time interval and end pva
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def test4(self):
        t0,p0,v0,p1,v1 = self.t0,self.p0,self.v0,self.p1,self.v1
        seg=Segment2ndDegree(t0,None,(p0,v0),(p1,v1)) # start + end positions + velocities
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def test5(self):
        t0,p0,v0,a,v1 = self.t0,self.p0,self.v0,self.a,self.v1
        seg=Segment2ndDegree(t0,None,(p0,v0,a),(None,v1)) # start pva + end velocity
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
    
    def test6(self):
        t1,p0,p1,v1,a = self.t1,self.p0,self.p1,self.v1,self.a
        seg=Segment2ndDegree(None,t1,p0,(p1,v1,a)) # end pva + start position
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def test7(self):
        t1,v0,p1,v1,a = self.t1,self.v0,self.p1,self.v1,self.a
        seg=Segment2ndDegree(None,t1,(None,v0),(p1,v1,a)) # end pva + start velocity
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def test8(self):
        t0,p0,p1,v0,a = self.t0,self.p0,self.p1,self.v0,self.a
        seg=Segment2ndDegree(t0,None,(p0,v0,a),p1) # start pva + end position
        assert_equal(seg.start(),self.start)
        assert_equal(seg.end(),self.end)
        
    def testover1(self):
        t0,t1,p0,v0,p1, a = self.t0,self.t1,self.p0,self.v0,self.p1, self.a*2 #double acceleration
        seg=Segment2ndDegree(t0,t1,(p0,v0,a),p1) # time interval, start pva, end position => adjust t1
        assert_equal(seg.start(),(p0,v0,a,0))
        assert_equal(seg.end()[0],self.p1)
        
    def testover2(self): #acceleration ramp
        t0,t1,p0,v0,v1, a = self.t0,self.t1,self.p0,self.v0,self.v1, self.a*2 #double acceleration
        seg=Segment2ndDegree(t0,t1,(p0,v0,a),(None,v1)) # time interval, start pva, v1= max vel => adjust t1
        assert_equal(seg.start(),(p0,v0,a,0))
        assert_equal(seg.end()[1],self.v1)
        assert_equal(seg.dt(),(self.t1-self.t0)/2) # double acceleration => half dt
        
    def test_segment2nd_degree(self):
        # assert_equal(expected, Segment2ndDegree(t0, t1, start, end))
        raise SkipTest 
    
class TestRamp:
    def test_ramp(self):
        pass #tested above and below

class TestTrapeze:
    def test_trapeze(self):
        assert_equal(trapeze(1,1,1),(1.0, 0.5, 1.0, 1.0, 0.5, 2.0))
        assert_equal(trapeze(2,1,1),(1.0, 0.5, 1.0, 2.0, 1.5, 3.0))
        assert_true(trapeze(0.1,12,10000)[-1]>trapeze(0.1,25,10000)[-1])


class TestSegmentTrapezoidalSpeed:
    def setup(self):
        self.t0, self.t1 = 1,2      
        self.p0, self.v0, self.a = -1,1,2
        self.start=(self.p0, self.v0, self.a, 0)
        self.p1, self.v1, self.a1 = 1,3,self.a
        self.end=(self.p1, self.v1, self.a1, 0)
    def test_segment_trapezoidal_speed(self):
        # assert_equal(expected, SegmentTrapezoidalSpeed(t0, p0, p1, a, T, vmax))
        raise SkipTest 

class TestSegment4thDegree:
    def setup(self):
        self.t0, self.t1 = 1,2      
        self.p0, self.v0, self.a0 = -1,1,0
        self.start=(self.p0, self.v0, self.a0, 0)
        self.p1, self.v1, self.a1 = 1,3,0
        self.end=(self.p1, self.v1, self.a1, 0)
        
    def test_segment4th_degree(self):
        seg=Segment4thDegree(self.t0, self.t1, self.start, self.end)
        assert_equal(seg.start()[:3],self.start[:3])  #ignore jerk
        assert_equal(seg.end()[:3],self.end[:3]) #ignore jerk
        assert_equal(seg((self.t0+self.t1)/2),(-0.3125, 2.0, 3.0, 0.0)) #truediv


if __name__ == "__main__":
    runmodule()
