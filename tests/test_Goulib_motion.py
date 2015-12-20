#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.motion import *
from math import pi
import os
path=os.path.dirname(os.path.abspath(__file__))

def pva_almost_equal(a,b,precision=6): # allow tests on Pt with 6 decimals precision
    list(map(lambda x:assert_almost_equal(x[0],x[1],precision),list(zip(a,b))))
    
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
    
class TestSegments:
    def test___init__(self):
        s1 = Segment2ndDegree(0,2,(0,0,2))
        s2 = Segment2ndDegree(2,4,(4.0,4.0,-2.0))
        segs = Segments([s1,s2])
        assert_equal(segs.t0, 0)
        assert_equal(segs.t1, 4)
        assert_equal(segs(2),(4.0,4.0,-2.0,0))
        assert_equal(segs.end(),(8.0,0.0,-2.0,0))
        s3 = Segment2ndDegree(4,6,(8,0,1))
        segs.add(s3)
        assert_equal(segs.t1, 6)
        assert_equal(segs.end(),(10.0,2.0,1.0,0)) 
        
    def test_add(self):
        s1 = Segment2ndDegree(0,2,(0,0,2))
        segs = Segments([s1])
        s2 = Segment2ndDegree(2,4,(4.0,4.0,-2.0))
        segs.add(s2)
        s3 = Segment2ndDegree(6,8,(8,0,1))
        segs.add(s3)  #should autoJoin
        assert_equal(segs.endTime(), 8)
        assert_equal(segs.end(),(10.0,2.0,1.0,0))
        assert_equal(len(segs.segments), 4, 'must have 4 segments: 3 added and one from the autojoin')
        
    def test_add_bug(self):
        s1 = SegmentsTrapezoidalSpeed(t0=0,p0=0,p3=1.8,a=0.5,vmax=1)
        s1.add(SegmentsTrapezoidalSpeed(t0=25.688904 , p0=1.8, p3=3.6, a=1))
        assert_equal(len(s1.segments), 5, 's1 is initally 3 + 1 autojoin + 1 Segments')
        
    def test_timeWhenPosBiggerThan(self):
        s1 = SegmentsTrapezoidalSpeed(t0=0,p0=0,p3=1.8,a=0.5,vmax=1)
        t = s1.timeWhenPosBiggerThan(1,resolution=0.01)
        assert_equal(t,2.01)
        
    def test_html(self):
        s1 = Segment2ndDegree(0,2,(0,0,2))
        s2 = Segment2ndDegree(2,4,(4.0,4.0,-2.0))
        segs = Segments([s1,s2])    
        assert_equal(segs.html(),'Segments starts=0 ends=4<br/>t=0.000000 (0.000000,0.000000,2.000000,0.000000) --> t=2.000000 (4.000000,4.000000,2.000000,0.000000)<br/>t=2.000000 (4.000000,4.000000,-2.000000,0.000000) --> t=4.000000 (8.000000,0.000000,-2.000000,0.000000)<br/>')
                

from Goulib.statemachine import StateMachine
class TestActuator:
    def test_move_horizontal(self):
        sm = StateMachine()
        a = Actuator(sm,V(1,'m/s'),V(1,'m/s^2'),name='m1',distPerTurn=V(1,'m'),mass=V(1,'kg'))
        # tests that if no move at the beginning, nothing crashes
        time = a.move(V(0,'m')).endTime()
        assert_equal(time,0.0)     
        
        time = a.move(V(3000,'mm')).endTime()
        assert_equal(a.segs.start(),(0.0, 0.0, 1.0, 0))
        assert_equal(a.segs.end(),(3.0, 0.0, -1.0, 0.0))
        assert_equal(time,4.0)
        a.move(V(0,'m'),acc=V(2,'m/s^2'))  #test overriding default acc
        assert_equal(a.segs.end(),(0.0, 0.0, 2.0, 0.0))
        #test that if no real move we get the same result
        a.move(V(0,'m'))
        assert_equal(a.segs.end(),(0.0, 0.0, 2.0, 0.0))  
        assert_equal(a.maxAbsAcc(),V(2,'m/s^2'))
        assert_equal(a.maxAbsSpeed(),V(1,'m/s'))
        assert_equal(a.maxRpm(),V(1,'1/s'))
        assert_almost_equal(a.maxTork()('N m'),0.31830988618379064)
        #assert_equal(a.varNames(),[])
        
    def test_move_vertical(self):
        sm = StateMachine()
        a = Actuator(sm,V(1,'m/s'),V(1,'m/s^2'),name='m1',distPerTurn=V(1,'m'),mass=V(1,'kg'),friction=V(0.25,'N'),angle=V(90,'deg'))
        a.move(V(-3,'m'))
        assert_almost_equal(a.maxForce()('N'),9.05665)
        a.move(V(0,'m'))
        assert_almost_equal(a.maxForce()('N'),11.05665)

class TestTimeDiagram:
        sm = StateMachine()
        a1 = Actuator(sm,V(1,'m/s'),V(1,'m/s^2'),name='m1',distPerTurn=V(1,'m'),mass=V(1,'kg'))    
        a2 = Actuator(sm,V(1,'m/s'),V(1,'m/s^2'),name='m1',distPerTurn=V(1,'m'),mass=V(1,'kg'))    
        a1.move(V(3000,'mm'))
        a2.move(V(3000,'mm'))
        assert_equal(sm.time,V(8.0,'s'))
        t = TimeDiagram([a1,a2])
        assert_equal(t.t0,0)
        assert_equal(t.t1,8)
        t.save('.\\tests\\results\\TimeDiagram.png',figsize=(20,20),dpi=600,linewidth=0.3)
        
                        
class TestSegmentPoly:
    @classmethod
    def setup_class(self):
        self.seg=SegmentPoly(1,2,[1,2,3])
        
    def test___init__(self):
        pass #tested above
        
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
        
    def test_save(self):
        self.seg.save(path+'/results/motion.SegmentPoly.png')

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
        assert_equal(seg.dt(),(self.t1-self.t0)/2.) # double acceleration => half dt
        
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
    def test_segmentTrapezoidalSpeed(self):
        trap = SegmentsTrapezoidalSpeed(t0=0,p0=0,p3=10.25,a=2.0,vmax=3.0,v0=1,v3=0) #start @t0 = 0
        assert_equal(trap.t1, 4.5)
        assert_equal(trap.end(),(10.25,0,-2,0))
        trap = SegmentsTrapezoidalSpeed(t0=1,p0=0,p3=10.25,a=2.0,vmax=3.0,v0=1,v3=0) #start @t0 != 0
        assert_equal(trap.endTime(), 5.5)
        assert_equal(trap.t0,1)
        assert_equal(trap.t1,5.5)
        assert_equal(trap.segments[0].t1, 2.0)
        assert_equal(trap.segments[1].t1, 4.0)
        assert_equal(trap.segments[2].t1, 5.5)
        assert_equal(trap.end(),(10.25,0,-2,0))
        trap = SegmentsTrapezoidalSpeed(t0=0,p0=0,p3=6.0,T=2.0,a=0.0,v0=2,v3=2) #time constraint with t0 = 0
        assert_equal(trap.end(),(6.0,2,-2,0))
        #time constraint with t0 != 0 not yet implemented
        
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
        assert_equal(seg((self.t0+self.t1)/2.),(-0.3125, 2.0, 3.0, 0.0)) #truediv


if __name__ == "__main__":
    runmodule()
