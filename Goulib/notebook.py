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
from .markup import tag
from .itertools2 import isiterable

def html(anything, sep=' '):
    try:
        return anything._repr_html_()
    except:
        pass
    
    if isiterable(anything): #iterable, but not a string
        try:
            return sep.join(html(a) for a in anything)
        except:
            pass
    
    try:
        return unicode(anything,'utf8') #to render accented chars correctly
    except:
        pass
    
    return str(anything)

sep=u' ' # Python2 doesn't allow named param after list of optional ones...

def h1(*args):
    display(HTML(tag('h1',html(args))))
    
def h2(*args):
    display(HTML(tag('h2',html(args))))
    
def h3(*args):
    display(HTML(tag('h3',html(args))))
    
def h(*args):
    display(HTML(html(args)))
    
def hinfo(*args):   
    display(HTML(tag('div',html(args),style="background-color:#337ab7;color:#ffffff")))
def hsuccess(*args):   
    display(HTML(tag('div',html(args),style="background-color:#5cb85c;color:#ffffff")))
def hwarning(*args):   
    display(HTML(tag('div',html(args),style="background-color:#f0ad4e;color:#ffffff")))
def herror(*args):   
    display(HTML(tag('div',html(args),style="background-color:#d9534f;color:#ffffff")))
