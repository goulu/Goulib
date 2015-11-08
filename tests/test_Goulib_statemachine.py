#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.StateMachine import *

class SM_Test(StateMachine):
    def state000(self):
        """titre
           actions
           --> 001: time out
        """
        print('000')
        self.time += 3 #elapsed time
    
    def state001(self):
        """titre
           actions
           --> 000: condition :to move
        """ 
        print('001')
        self.time += 7


class TestSM_test:
    
    def test___init__(self):
        sm = SM_Test()
        print(sm.states)
    
    def test__html_repr_(self):
        sm = SM_Test()
        assert_equal(sm._repr_html_(), '<table border="1"><caption>SM_Test</caption><tr><td>0</td><td>titre</td><td>actions</td></tr><tr><td>1</td><td>titre</td><td>actions</td></tr></table>')
        

    def test_run(self):
        sm = SM_Test()
        sm.run(start=0,maxState=4)
        assert_equal(sm.log, [(0, 0), (1, 3), (0, 10), (1, 13)])
        
        
from Goulib.motion import *


class GripperSM(StateMachine):
    def __init__(self):
        StateMachine.__init__(self)
        self.m20 = Actuator(self,1,1)
        self.m21 = Actuator(self,1,1)
        self.m22 = Actuator(self,1,1)
        self.m23 = Actuator(self,1,1)
        self.m24 = Actuator(self,1,1)
        self.m25 = Actuator(self,1,1)
        self.m26 = Actuator(self,1,1)
        
        
    def state000(self):
        """init
        --> 10:
        """
        return 0
    
    def state010(self):
        """GripperAbovePile
           enterNewPile
        --> 20: pc == PileEntered
        """
        display(self.m25.move(2))
        return 5
    
    def state020(self):
        """LowerCraneToPileTop
        M23 ramp according to
        -->30:gripper in separation position
        """
        
        
    def state030(self):
        """close jogger
        M20+M21 controlled in position + torque control??
        --> 40:jogger closed
        """
        
    def state040(self):
        """BatchPushSeparator
        Ya1=1
        --> 50:timeout == 1sec
        """
        
    def state050(self):
        """LiftCraneToSeparate
        M23 ramp
        --> 60: in pos """
        
    def state060(self):
        """IntroduceFork
        M22 ramp
        --> 70: M22 in pos """
        
    def state70(self):
        """LiftCraneAbovePile
        M23 ramp
        --> 80: M23 in pos """
        
    def state80(self):
        """ShiftCraneAboveDepositTable
        M24 ramp
        --> 90: M24 in pos"""
        
    def state90(self):
        """LowerCraneToDepositTable
        M23 ramp
        -->100: M23 in pos """
        
    def state100(self):
        """RemoveFork
        M22 ramp
        -->110: M22 in pos """
        
    def state110(self):
        """LiftGripperAbovePile
        M23 ramp
        -->120: M23 in pos"""
        
    def state120(self):
        """ShiftCraneOnTopOfPile
        M24 ramp
        --> 10: M24 in pos """
        
class TestGripper:
    def test_001(self):
        gr=GripperSM()
        gr.run(maxState=5)
        print(gr.log)        
 
if __name__ == "__main__":
    runmodule()           