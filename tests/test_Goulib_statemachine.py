#!/usr/bin/env python
# coding: utf8
from nose.tools import assert_equal
from nose import SkipTest
#lines above are inserted automatically by pythoscope. Line below overrides them
from Goulib.tests import *
from Goulib.statemachine import *
from Goulib.motion import Actuator

class SM_Test(StateMachine):
    def __init__(self,simulation):
        StateMachine.__init__(self,simulation)
        self.m1 = Actuator(self,V(1,'m/s'),V(1,'m/s^2'),'m1')
        
    def state000(self):
        """titre
           actions
           --> 001: time out
        """
        logging.debug('000')
        self.m1.move(V(2,'m'))
        self.simulation.aGlobalVar = 'toto'
    
    def state001(self):
        """titre
           actions
           --> 000: condition :to move
        """ 
        logging.debug('001')
        self.wait(self.time +V(7,'s'))
        assert self.simulation.aGlobalVar == 'toto' 


class TestSM_test:
    
    def test___init__(self):
        simulation = Simulation()
        sm = SM_Test(simulation)
        logging.debug(sm.states)
    
    def test__html_repr_(self):
        simulation = Simulation()
        sm = SM_Test(simulation)
        assert_equal(sm._repr_html_(), '<table border="1"><caption>SM_Test</caption><tr><td>0</td><td>titre</td><td>actions</td></tr><tr><td>1</td><td>titre</td><td>actions</td></tr></table>')
        

    def test_run(self):
        simulation = Simulation()
        sm = SM_Test(simulation)
        sm.run(start=0,maxSteps=4)
        assert_equal(sm.lastExitTime(0), V(10,'s'))
        assert_equal(sm(-1),None)
        assert_equal(sm(2),0)
        assert_equal(sm(3),1)
        assert_equal(sm(4),1)
        assert_equal(sm(10),1)
        assert_equal(sm(11),1)
        
class TestTimeMarker:
    def test_all(self):
        tm = TimeMarker('test_tm')
        assert_equal(tm()('s'),-float('inf'))
        tm.set(V(3,'s'))
        assert_equal(tm()('s'),3)
        tm.set(V(5,'s'))
        assert_equal(tm()('s'),5)
        assert_equal(str(tm),'test_tm 3.000000[s] 5.000000[s]')
        
 
if __name__ == "__main__":
    runmodule()           