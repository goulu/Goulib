#!/usr/bin/env python
# coding: utf8__author__ = "Marc Nicole"

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"

from IPython.display import display, HTML
import inspect
from Goulib.units import V

from graphviz import Digraph

class StateDiagram(Digraph):
    """
    helper to write State Diagrams graph in iPython notebook
    This library uses Graphviz that has to be installed separately (http://www.graphviz.org/)
    it uses graphviz python libraray http://graphviz.readthedocs.org/en/latest/
    """
    def state(self,number,descr,actions,transitions):
        """ :parameter number: the state number (int)
            :parameter descr: a string describing the state
            :parameter actions: a string describing the various actions. use <br/> to make a new line
            :parameter transitions: a dict of {<new_state>:"condition",...}
        """
        html= '<<table border="0" cellspacing="0" cellborder="1"><tr><td>'+str(number)+'</td><td>'+descr+'</td></tr><tr><td colspan="2" align="left">'+actions+'</td></tr></table>>'
        self.node(str(number),html,shape='none')
        for newState in transitions:
            self.edge(str(number),str(newState),transitions[newState])

maxDiff = None

class StateMachine:
    def __init__(self):
        self.displayNotebook = False
        self.time = V(0.0,'s')
        self.name = self.__class__.__name__
        self.states = {}
        funcs = inspect.getmembers(self,predicate=inspect.ismethod)
        for (fname,f) in funcs:
            if fname[0:5]=='state':
                stateNumber = int(fname[5:])
                self.states[stateNumber] = {'action':f}
                self.parseDoc(stateNumber,f)

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
                
        
    def run(self,start=0,stops=[],maxState=100,maxTime=V(100,'s'),displayNotebook=False):
        """ runs the behavioral simulation 
            :params start: is the starting state of the simulation
            :params stops: a list of states that will stop the simulation (after having simulated this last state)
            :params maxState: is the number of states being evaluated before the end of simulation
            :params maxTime: is the virtual time at which the simulation ends_in_comment_or_string
        """
        
        self.time = V(0,'s')
        self.displayNotebook = displayNotebook
        currentState = start
        self.log = []
        while len(self.log) < maxState and self.time < maxTime:
            self.log.append((currentState,self.time))
            if displayNotebook:
                display(HTML('<h3>{0} {1}</h3>'.format(currentState,self.states[currentState]['title'])))
            self.next = self.states[currentState]['transitions'][0][0]  #by default the next state is the first transition
            self.states[currentState]['action']()
            if currentState in stops:
                break
            currentState = self.next 
            
                
        