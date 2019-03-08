#!/usr/bin/env python
# coding: utf8
"""
python function to write html texts and values inline within a Notebook

Advantagously replaces "print" since this is more powerfull and has no compatibility issues between python 2 and 3

exemple: h('the result is',time,'[ms]')

"""

from __future__ import print_function

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2016, Philippe Guglielmetti"
__credits__= [""]
__license__ = "LGPL"

import logging

from IPython.display import display, HTML, Math
from .markup import tag
from .itertools2 import isiterable


def html(obj, sep=None):
    try:
        return obj._repr_html_()
    except AttributeError:
        pass #skip logging.error

    if sep is None:
        sep=' '
        bra,ket='',''
    else:
        if isinstance(obj,dict):
            res=',\n'.join("%s:%s"%(html(k),html(v)) for k,v in obj.items())
            return '{%s}'%res
        elif isinstance(obj,list):
            bra,ket='[',']'
        else:
            bra,ket='(',')'

    if isiterable(obj): #iterable, but not a string
        return bra+sep.join(html(a,sep=',') for a in obj)+ket

    return str(obj)

def h1(*args):
    display(HTML(tag('h1',html(args))))

def h2(*args):
    display(HTML(tag('h2',html(args))))

def h3(*args):
    display(HTML(tag('h3',html(args))))

def h4(*args):
    display(HTML(tag('h4',html(args))))

def h(*args):
    display(HTML(html(args)))

#redefine "print" for notebooks ...
try: #http://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    get_ipython #is defined from within IPython (notebook)
except:
    pass
else:
    pass #for pythoscope
    # print = h # this is ok in Python 3, but not before

def hinfo(*args):
    display(HTML(tag('div',html(args),style="background-color:#337ab7;color:#ffffff")))
def hsuccess(*args):
    display(HTML(tag('div',html(args),style="background-color:#5cb85c;color:#ffffff")))
def hwarning(*args):
    display(HTML(tag('div',html(args),style="background-color:#f0ad4e;color:#ffffff")))
def herror(*args):
    display(HTML(tag('div',html(args),style="background-color:#d9534f;color:#ffffff")))

def latex(obj):
    """ to force LaTeX representation """
    return Math(obj.latex())