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
        if isinstance(obj,HTML):
            notebookFile.write(obj._repr_html_())
        elif isinstance(obj,Image):
            b = b64encode(obj.data)
            c = b.decode('utf-8')
            notebookFile.write('<img src="data:image/png;base64,'+c+'">')
        elif isinstance(obj,SVG):
            notebookFile.write(obj.data)
        else:
            notebookFile.write('<h1 style="color:red;">****** {0} ******</h1>'.format(type(obj).__name__))
            
            
    
    
sep=' ' # Python2 doesn't allow named param after list of optional ones...

def h1(*args):
    display(HTML('<h1>'+sep.join(str(a) for a in args)+'</h1>'))
    
def h2(*args):
    display(HTML('<h2>'+sep.join(str(a) for a in args)+'</h2>'))
    
def h3(*args):
    display(HTML('<h3>'+sep.join(str(a) for a in args)+'</h3>'))
    
def h(*args):
    display(HTML(sep.join(str(a) for a in args))) 
    
def hinfo(*args):   
    display(HTML('<div style="background-color:#337ab7;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def hsuccess(*args):   
    display(HTML('<div style="background-color:#5cb85c;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def hwarning(*args):   
    display(HTML('<div style="background-color:#f0ad4e;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   

def herror(*args):   
    display(HTML('<div style="background-color:#d9534f;color:#ffffff">'+sep.join(str(a) for a in args)+'</div>'))   
    
