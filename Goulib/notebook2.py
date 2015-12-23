#!/usr/bin/env python
# coding: utf8
"""
python function to write html texts and values inline within a Notebook

Advantagously replaces "print" since this is more powerfull and has no compatibility issues between python 2 and 3

exemple: h('the result is',time,'[ms]')

"""
from sphinx.util import FilenameUniqDict

__author__ = "Marc Nicole"
__copyright__ = "Copyright 2015, Marc Nicole"
__credits__= [""]
__license__ = "LGPL"


from base64 import b64encode
from IPython.display import HTML,Image,SVG
import IPython,subprocess
from Goulib.statemachine import StateDiagram


notebookFile = None
notebookFilename = ''



def setNotebookToFile(filename):
    global notebookFile, notebookFilename
    notebookFilename = filename
    notebookFile = open(filename,'w',encoding='utf-8')
    notebookFile.write("""<!DOCTYPE html>
    <html>
      <head>
         <style>
         body {font-family:"Segoe UI",Arial,sans-serif;}
         </style>
      </head>
      <body>
    """)
    
def closeNotebook():
    global notebookFile
    if notebookFile:
        notebookFile.write("""  </body>
          </html>""")
        notebookFile.close()
        subprocess.Popen([notebookFilename],shell=True)

def display(obj):
    global file
    if notebookFile is None:
        IPython.display.display(obj)
    else:
        if getattr(obj,'_repr_svg_',None):
            notebookFile.write(obj._repr_svg_())
        elif isinstance(obj,Image):
            b = b64encode(obj.data)
            c = b.decode('utf-8')
            notebookFile.write('<img src="data:image/png;base64,'+c+'">')
        elif getattr(obj,'_repr_html_',None):
            notebookFile.write(obj._repr_html_())
        else:
            notebookFile.write('<h1 style="color:red;">****** {0} ******</h1>'.format(type(obj).__name__))
            notebookFile.write('<p>'+str(obj)+'</p>')
            
            
    
    
sep=' ' # Python2 doesn't allow named param after list of optional ones...

MAXVERBOSITY = {'h1':True,'h2':True,'h3':True,'h':True,'hinfo':True,'hsuccess':True,'hwarning':True,'herror':True}
SILENT = {'h1':False,'h2':False,'h3':False,'h':False,'hinfo':False,'hsuccess':False,'hwarning':False,'herror':False}

verbosity = MAXVERBOSITY

hasErrors = False
hasWarning = False

def h1(*args):
    if verbosity['h1']:
        display(HTML('<h1>'+sep.join(str(a) for a in args)+'</h1>'))
    
def h2(*args):
    if verbosity['h2']:
        display(HTML('<h2>'+sep.join(str(a) for a in args)+'</h2>'))
    
def h3(*args):
    if verbosity['h3']:
        display(HTML('<h3>'+sep.join(str(a) for a in args)+'</h3>'))
    
def h(*args):
    if verbosity['h']:
        display(HTML(sep.join(str(a) for a in args))) 
    
def hinfo(*args):   
    if verbosity['hinfo']:
        display(HTML('<div style="background-color:#337ab7;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def hsuccess(*args):   
    if verbosity['hsuccess']:
        display(HTML('<div style="background-color:#5cb85c;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def hwarning(*args):
    global hasWarnings
    hasWarnings = True   
    if verbosity['hwarning']:
        display(HTML('<div style="background-color:#f0ad4e;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def herror(*args):
    global hasErrors
    hasErrors = True   
    if verbosity['herror']:
        display(HTML('<div style="background-color:#d9534f;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   
    
