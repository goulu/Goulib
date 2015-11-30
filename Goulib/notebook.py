#!/usr/bin/env python
# coding: utf8
"""
python function to write html texts and values inline within a Notebook

Advantagously replaces "print" since this is more powerfull and has no compatibility issues between python 2 and 3

exemple: h('the result is',time,'[ms]')

"""

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"

from IPython.display import display, HTML

sep=' ' # Python2 doesn't allow named param after list of optional ones...

def h1(*args):
    display(HTML('<h1>'+sep.join(str(a) for a in args)+'</h1>'))
    
def h2(*args):
    display(HTML('<h2>'+sep.join(str(a) for a in args)+'</h2>'))
    
def h3(*args):
    display(HTML('<h3>'+sep.join(str(a) for a in args)+'</h3>'))
    
def h(*args):
    display(HTML(sep.join(str(a) for a in args)))       