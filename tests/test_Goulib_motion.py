from nose.tools import assert_equal, assert_almost_equal
from nose import SkipTest
from Goulib.motion import *

def assert_almost_equal(a,b,precision=6): # allow tests on Pt with 6 decimals precision
    map(lambda x:nose.tools.assert_almost_equal(x[0],x[1],precision),zip(a,b))
    
class TestSegmentPoly:
    def setup(self):
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
        assert_almost_equal(self.seg(2-1e-12),(6,8,6,0))
        assert_equal(self.seg(2),(0,0,0,0))
        
class TestSegment2ndDegree:
    def setup(self):
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
        
    def testover1(self):
        t0,t1,p0,v0,p1, a = self.t0,self.t1,self.p0,self.v0,self.p1, self.a*2 #double acceleration
        seg=Segment2ndDegree(t0,t1,(p0,v0,a),p1) # time interval, start pva, end position => adjust t1
        assert_equal(seg.start(),(p0,v0,a,0))
        assert_equal(seg.end()[0],self.p1)
        assert_equal(seg.dt(),(self.t1-self.t0)/2.0) # double acceleration => half dt
        
    def testover2(self):
        t0,t1,p0,v0,v1, a = self.t0,self.t1,self.p0,self.v0,self.v1, self.a*2 #double acceleration
        seg=Segment2ndDegree(t0,t1,(p0,v0,a),(None,v1)) # time interval, start pva, max vel => adjust t1
        assert_equal(seg.start(),(p0,v0,a,0))
        assert_equal(seg.end()[1],self.v1)
        assert_equal(seg.dt(),(self.t1-self.t0)/2.0) # double acceleration => half dt
        
if __name__ == "__main__":
    import nose
    nose.runmodule()
