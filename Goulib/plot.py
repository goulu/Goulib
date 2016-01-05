#!/usr/bin/env python
# coding: utf8
"""
plotable rich object display on IPython notebooks 
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

#import matplotlib and set backend once for all
import matplotlib, os, sys, logging, six

if os.getenv('TRAVIS'): # are we running https://travis-ci.org/ automated tests ?
    matplotlib.use('Agg') # Force matplotlib  not to use any Xwindows backend
elif sys.gettrace(): #http://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger
    matplotlib.use('Agg') #because 'QtAgg' crashes python while debugging
else:
    pass
    # matplotlib.use('pdf') #for high quality pdf, but doesn't work for png, svg ...
    
logging.info('matplotlib backend is %s'%matplotlib.get_backend())

from . import itertools2

class Plot(object):
    """base class for plotable rich object display on IPython notebooks
    inspired from http://nbviewer.ipython.org/github/ipython/ipython/blob/3607712653c66d63e0d7f13f073bde8c0f209ba8/docs/examples/notebooks/display_protocol.ipynb
    """
    
    def _plot(self, ax, **kwargs):
        """abstract method, must be overriden
        
        :param ax: `matplotlib.axis` 
        :return ax: `matplotlib.axis` after plot
        """
        raise NotImplementedError('objects derived from plot.PLot must define a _plot method')
        return ax
    
    def render(self, fmt='svg', **kwargs):
        return render([self],fmt, **kwargs) # call global function
    
    def save(self,filename,**kwargs):
        return save([self],filename, **kwargs) # call global function
    
    def _repr_png_(self,**kwargs):
        try:
            return self.render(fmt='png',**kwargs)
        except: #maybe this object is not plotable
            return None

    def _repr_svg_(self,**kwargs):
        try:
            return self.render(fmt='svg',**kwargs).decode('utf-8')
        except: #maybe this object is not plotable
            return None
    
    def png(self,**kwargs):
        from IPython.display import Image
        return Image(self._repr_png_(**kwargs), embed=True)
    
    def svg(self,**kwargs):
        from IPython.display import SVG
        return SVG(self._repr_svg_(**kwargs))
    
    def plot(self,**kwargs):
        """ renders on IPython Notebook
        (alias to make usage more straightforward)
        """
        return self.svg(**kwargs)
    
    
def render(plotables, fmt='svg', **kwargs):
    """renders several Plot objects"""
    import matplotlib.pyplot as plt

    #extract optional arguments used for rasterization
    printargs,kwargs=itertools2.dictsplit(
        kwargs,
        ['dpi','transparent','facecolor','background','figsize']
    )
    
    ylim=kwargs.pop('ylim',None)
    xlim=kwargs.pop('xlim',None)
    title=kwargs.pop('title',None)
    
    fig, ax = plt.subplots()
    
    labels=kwargs.pop('labels',[None]*len(plotables))
    offset=kwargs.pop('offset',0) #slightly shift the points to make superimposed curves more visible
        
    for i,obj in enumerate(plotables):
        if labels[i] is None:
            labels[i]=str(obj)
        if not title:
            try:
                title=obj._repr_latex_()
            except:
                title=labels[i]
        ax = obj._plot(ax, label=labels[i], offset=i*offset, **kwargs)       
    
    if ylim: plt.ylim(ylim)
    if xlim: plt.xlim(xlim)
    

    if title: ax.set_title(title) 
    if len(labels)>1:
        ax.legend()
        
    from io import BytesIO
    output = BytesIO()
    fig.savefig(output, format=fmt, **printargs)
    data=output.getvalue()
    plt.close(fig)
    return data

def png(plotables, **kwargs):
    from IPython.display import Image
    return Image(render(plotables,'png',**kwargs), embed=True)
    
def svg(plotables, **kwargs):
    from IPython.display import SVG
    return SVG(render(plotables,'svg',**kwargs))

plot=svg

def save(plotables,filename,**kwargs):
    ext=filename.split('.')[-1].lower()
    kwargs.setdefault('dpi',600) #force good quality
    return open(filename,'wb').write(render(plotables, ext,**kwargs))

    