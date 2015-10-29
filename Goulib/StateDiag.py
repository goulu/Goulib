#!/usr/bin/env python
# coding: utf8
"""
helpers to write State Diagrams graph in iPython notebook

This library uses Graphviz that has to be installed separately (http://www.graphviz.org/)

it uses graphviz python libraray http://graphviz.readthedocs.org/en/latest/
"""

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"

from graphviz import Digraph
class StateDiagram(Digraph):
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