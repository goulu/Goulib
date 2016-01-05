#!/usr/bin/env python
# coding: utf8

"""
state machines with graph representation
"""
from scipy.stats.mstats_basic import argstoarray

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"
#test rebase

import inspect

from Goulib.units import V
from Goulib.piecewise import Piecewise
from graphviz import Digraph


class StateDiagram(Digraph):
    """
    helper to write State Diagrams graph in iPython notebook
    This library uses Graphviz that has to be installed separately (http://www.graphviz.org/)
    """
    def state(self,number,descr,actions,transitions):
        """ :parameter number: the state number (int)
            :parameter descr: a string describing the state
            :parameter actions: a string describing the various actions. use <br/> to make a new line
            :parameter transitions: a array of tuple (<new_state>,"condition")
        """
        html= '<<table border="0" cellspacing="0" cellborder="1"><tr><td>'+str(number)+'</td><td>'+descr+'</td></tr><tr><td colspan="2" align="left">'+actions+'</td></tr></table>>'
        self.node(str(number),html,shape='none')
        for transition in transitions:
            self.edge(str(number),str(transition[0]),transition[1]) #coucou

maxDiff = None


def noPrint(*args):
    pass

class Simulation:
    """ all simulation should derive from this class that has some helper
    """
    def __init__(self):
        self.h1           = noPrint
        self.h2           = noPrint
        self.h3           = noPrint
        self.h            = noPrint
        self.hinfo        = noPrint
        self.hsuccess     = noPrint
        self.hwarning     = noPrint
        self.herror       = noPrint
        self.displayState = noPrint
        self.displayPlot  = noPrint
        
    def setOutput(self,h1=noPrint,h2=noPrint,h3=noPrint,h=noPrint,hinfo=noPrint,hsuccess=noPrint,hwarning=noPrint,herror=noPrint,displayState=noPrint,displayObj=noPrint,displayPlot=noPrint):
        self.h1        = h1
        self.h2        = h2
        self.h3        = h3
        self.h         = h
        self.hinfo     = hinfo
        self.hsuccess  = hsuccess
        self.hwarning  = hwarning
        self.herror    = herror
        self.displayState = displayState
        self.displayPlot  = displayPlot

#------------ for the logs -----------
class EventLog:
    def log(self,stateMachine):
        stateMachine.log.append((stateMachine.time('s'),self))
        
class StateChangeLog(EventLog):
    def __init__(self,newState):
        self.newState = newState
        
class WaitLog(EventLog):
    def __init__(self,untilTime,waitForWhat):
        self.untilTime = untilTime
        self.waitForWhat = waitForWhat
        
class TooLateLog(EventLog):
    def __init__(self,pastTime,missedWhat):
        self.pastTime = pastTime
        self.missedWhat = missedWhat
            
#----------------------------------------            
class TimeMarker:
    def __init__(self,name):
        self.name = name
        self.markers = [-float('inf')]
        
    def set(self,time):
        self.markers.append(time('s'))
        
    def __call__(self):
        return V(self.markers[-1],'s')
    
    def __repr__(self):
        s = self.name
        for t in self.markers[1:]:
            s += ' %f[s]' % t
        return s
     
     
class StateMachine:
    def __init__(self,simulation=None,name=None,background_color="#F5ECCE"):
        self.displayMove = False
        self.time = V(0.0,'s')
        self.background_color = background_color
        self.simulation = simulation
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.states = {}
        funcs = inspect.getmembers(self,predicate=inspect.ismethod)
        for (fname,f) in funcs:
            if fname[0:5]=='state':
                stateNumber = int(fname[5:])
                self.states[stateNumber] = {'action':f}
                self.parseDoc(stateNumber,f)
        self.reset()
        
    def reset(self):
        """ where all actuators should be declared and other variables"""
        self.__reset__()
        self.log = []
        self.hasErrors = False
        self.hasWarnings = False
        
    def __reset__(self):
        pass

    def parseDoc(self,state,f):
        doclines = f.__doc__.split('\n')
        self.states[state]['title'] = doclines[0]
        self.states[state]['transitions'] = []
        actions = []
        for dl in doclines[1:]:
            d = dl.strip()
            if d[0:3] == '-->':
                t = d[3:].split(':',1)
                newState = int(t[0])
                self.states[state]['transitions'].append((newState,t[1]))
            elif d != '':
                actions.append(d)
        self.states[state]['actions']=actions
            
    def __call__(self,time):
        """ find the state at time.  time must be in seconds """
        for t,event in reversed(self.log):
            if t <= time and isinstance(event,StateChangeLog):
                return event.newState
        return None
            
        
            
    def _repr_html_(self):
        html = '<table border="1"><caption>'+self.name+'</caption>'
        for state in self.states:
            html += '<tr><td>'+str(state)+'</td><td>'+self.states[state]['title']+'</td><td>'+'<br/>'.join(self.states[state]['actions'])+'</td></tr>'
            
        html += '</table>'
        return html
                
    def displayGraph(self):
        from Goulib.notebook2 import display         
        graph = StateDiagram(self.name)
        for state in self.states:
            graph.state(state,self.states[state]['title'], '<br/>'.join(self.states[state]['actions']), self.states[state]['transitions'])
        display(graph)
        
    def checkOnTimeAndWait(self,time,what):
        """ checks that the self.time < time
        if this is not the case an error will be logged with the message"""
        if self.time > time:
            TooLateLog(time,what).log(self)
            self.herror(what,'is too late: ',self.time, 'instead of',time)
        else:
            WaitLog(time,what).log(self)
            self.hsuccess('waits for ',what,'from',self.time,'to',time)
            self.time = time
            
    def wait(self,time,cause='unknown cause'):
        if time > self.time:
            self.time = time
            #TODO rethink how to display info .....hinfo(self.name+' waits for ',cause)
            
    def run(self,start=0,stops=[],startTime=None,maxSteps=100000,maxTime=V(1000,'s')):
        """ runs the behavioral simulation 
            :params start: is the starting state of the simulation
            :params stops: a list of states that will stop the simulation (after having simulated this last state)
            :params startTime: a time to start this run if None takes self.time
            :params maxState: is the number of states being evaluated before the end of simulation
            :params maxTime: is the virtual time at which the simulation ends_in_comment_or_string
            :params displayStates: at every new state, display the state in Notebook as well as the time when entered
            :params displayMove: if True, every actuator.move displays the graph of the move
            
            returns the time when run finishes
        """
        if startTime:
            self.time = startTime
        currentState = start
        steps = 0
        while steps < maxSteps and self.time < maxTime:
            StateChangeLog(currentState).log(self)
            self.simulation.displayState(self.name,currentState,self.states[currentState]['title'],self.time,self.background_color)
            self.next = self.states[currentState]['transitions'][0][0]  #by default the next state is the first transition
            self.states[currentState]['action']()
            if currentState in stops:
                break
            currentState = self.next 
            steps +=1
        return self.time
    
    def hinfo(self,*args):
        self.simulation.hinfo(*args)
            
    def hsuccess(self,*args):
        self.simulation.hsuccess(*args)
            
    def hwarning(self,*args):
        self.simulation.hwarning(*args)
        self.hasWarnings = True
            
    def herror(self,*args):
        self.simulation.herror(*args)
        self.hasErrors = True
        
            
    def lastExitTime(self,state):
        last = -float('inf')
        for i in range(len(self.log)):
            event = self.log[i][1]
            if isinstance(event,StateChangeLog) and event.newState==state:
                last = self.log[i+1][0]
        return V(last,'s')
    
    def display(self,fromTime=None,toTime=None):
        p = Piecewise(init=self.log)
        from IPython.display import display,HTML
        display(HTML('<h4>{0}</h4>'.format(self.name)))
        if fromTime is not None:
            fromTime = fromTime('s')
        if toTime is not None:
            toTime = toTime('s')
        display(p.svg(xlim=(fromTime,toTime)))
        

        
        