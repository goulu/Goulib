#!/usr/bin/env python
# coding: utf8

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"


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
            
            
    def _repr_html_(self):
        html = '<table border="1"><caption>'+self.name+'</caption>'
        for state in self.states:
            html += '<tr><td>'+str(state)+'</td><td>'+self.states[state]['title']+'</td><td>'+'<br/>'.join(self.states[state]['actions'])+'</td></tr>'
            
        html += '</table>'
        return html
                
    def displayGraph(self):
        from IPython.display import display         
        graph = StateDiagram(self.name)
        for state in self.states:
            graph.state(state,self.states[state]['title'], '<br/>'.join(self.states[state]['actions']), self.states[state]['transitions'])
        display(graph)
        
    def wait(self,time):
        if time > self.time:
            self.time = time
            
    def run(self,start=0,stops=[],startTime=V(0,'s'),maxSteps=100000,maxTime=V(1000,'s'),displayStates=False,displayMove=False):
        """ runs the behavioral simulation 
            :params start: is the starting state of the simulation
            :params stops: a list of states that will stop the simulation (after having simulated this last state)
            :params startTime: a time to start this run
            :params maxState: is the number of states being evaluated before the end of simulation
            :params maxTime: is the virtual time at which the simulation ends_in_comment_or_string
            :params displayStates: at every new state, display the state in Notebook as well as the time when entered
            :params displayMove: if True, every actuator.move displays the graph of the move
        """
        
        self.time = startTime
        self.displayMove = displayMove
        self.displayStates = displayStates
        currentState = start
        steps = 0
        while steps < maxSteps and self.time < maxTime:
            self.log.append((self.time.magnitude,currentState))
            if displayStates:
                from IPython.display import display,HTML
                display(HTML('<div style="padding:5px;background-color:'+self.background_color+';"><h3>{0}={1} {2}</h3>{3:f}</div>'.format(self.name,currentState,self.states[currentState]['title'],self.time)))
            self.next = self.states[currentState]['transitions'][0][0]  #by default the next state is the first transition
            self.states[currentState]['action']()
            if currentState in stops:
                break
            currentState = self.next 
            steps +=1
            
    def lastExitTime(self,state):
        last = -float('inf')
        for i in range(len(self.log)):
            if self.log[i][1]==state:
                last = self.log[i+1][0]
        return V(last,'s')
    
    def display(self,fromTime=None,toTime=None):
        p = Piecewise(init=self.log)
        from IPython.display import display,HTML
        display(HTML('<h4>{0}</h4>'.format(self.name)))
        if fromTime is not None:
            fromTime = fromTime('s')
        if toTime is not None:
            toTime =toTime('s')
        display(p.svg(xlim=(fromTime,toTime)))
        

    
        