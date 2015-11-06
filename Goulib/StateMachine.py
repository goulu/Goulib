'''
Created on 5 nov. 2015

@author: Marc
'''

from IPython.display import display, HTML, Image
import inspect

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
                
        
