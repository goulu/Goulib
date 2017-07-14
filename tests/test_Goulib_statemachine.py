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
        
 
    def test___call__(self):
        # time_marker = TimeMarker(name)
        # assert_equal(expected, time_marker.__call__())
        raise SkipTest # implement your test here

    def test___init__(self):
        # time_marker = TimeMarker(name)
        raise SkipTest # implement your test here

    def test___repr__(self):
        # time_marker = TimeMarker(name)
        # assert_equal(expected, time_marker.__repr__())
        raise SkipTest # implement your test here

    def test_set(self):
        # time_marker = TimeMarker(name)
        # assert_equal(expected, time_marker.set(time))
        raise SkipTest # implement your test here

class TestStateDiagram:
    def test_state(self):
        # state_diagram = StateDiagram()
        # assert_equal(expected, state_diagram.state(number, descr, actions, transitions))
        raise SkipTest 

class TestStateMachine:
    def test___init__(self):
        # state_machine = StateMachine(simulation, name, background_color)
        raise SkipTest 

    def test___reset__(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.__reset__())
        raise SkipTest 

    def test_display(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.display(fromTime, toTime))
        raise SkipTest 

    def test_displayGraph(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.displayGraph())
        raise SkipTest 

    def test_lastExitTime(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.lastExitTime(state))
        raise SkipTest 

    def test_parseDoc(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.parseDoc(state, f))
        raise SkipTest 

    def test_reset(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.reset())
        raise SkipTest 

    def test_run(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.run(start, stops, startTime, maxSteps, maxTime, displayStates, displayMove))
        raise SkipTest 

    def test_wait(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.wait(time))
        raise SkipTest 

    def test___call__(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.__call__(time))
        raise SkipTest # implement your test here

    def test_checkOnTimeAndWait(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.checkOnTimeAndWait(time, what))
        raise SkipTest # implement your test here

    def test_herror(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.herror(*args))
        raise SkipTest # implement your test here

    def test_hinfo(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.hinfo(*args))
        raise SkipTest # implement your test here

    def test_hsuccess(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.hsuccess(*args))
        raise SkipTest # implement your test here

    def test_hwarning(self):
        # state_machine = StateMachine(simulation, name, background_color)
        # assert_equal(expected, state_machine.hwarning(*args))
        raise SkipTest # implement your test here

class TestNoPrint:
    def test_no_print(self):
        # assert_equal(expected, noPrint(*args))
        raise SkipTest # implement your test here

class TestSimulation:
    def test___init__(self):
        # simulation = Simulation()
        raise SkipTest # implement your test here

    def test_setOutput(self):
        # simulation = Simulation()
        # assert_equal(expected, simulation.setOutput(h1, h2, h3, h, hinfo, hsuccess, hwarning, herror, displayState, displayObj, displayPlot))
        raise SkipTest # implement your test here

class TestEventLog:
    def test_log(self):
        # event_log = EventLog()
        # assert_equal(expected, event_log.log(stateMachine))
        raise SkipTest # implement your test here

class TestStateChangeLog:
    def test___init__(self):
        # state_change_log = StateChangeLog(newState)
        raise SkipTest # implement your test here

class TestWaitLog:
    def test___init__(self):
        # wait_log = WaitLog(untilTime, waitForWhat)
        raise SkipTest # implement your test here

class TestTooLateLog:
    def test___init__(self):
        # too_late_log = TooLateLog(pastTime, missedWhat)
        raise SkipTest # implement your test here

if __name__ == "__main__":
    runmodule()           