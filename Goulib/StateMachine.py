#!/usr/bin/env python
# coding: utf8
'''
Created on 5 nov. 2015

@author: Marc
''' 

from IPython.display import display, HTML, Image
import inspect
from bokeh.state import State
from IPython.core.inputtransformer import ends_in_comment_or_string

maxDiff = None

class StateMachine:
    def __init__(self):
        self.time = 0.0
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
                
        
    def run(self,start=0,stops=[],maxState=100,maxTime=100,displayNotebook=False):
        """ runs the behavioral simulation 
            :params start: is the starting state of the simulation
            :params stops: a list of states that will stop the simulation (after having simulated this last state)
            :params maxState: is the number of states being evaluated before the end of simulation
            :params maxTime: is the virtual time at which the simulation ends_in_comment_or_string
        """
        
        self.time = 0
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
            
                
        