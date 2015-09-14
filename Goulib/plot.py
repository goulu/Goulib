# -*- coding: utf-8 -*-
"""
plot utilities
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2015, Philippe Guglielmetti"
__credits__ = []
__license__ = "LGPL"

#import matplotlib and set backend once for all
import matplotlib, os, sys, logging

if os.getenv('TRAVIS'): # are we running https://travis-ci.org/ automated tests ?
    matplotlib.use('Agg') # Force matplotlib  not to use any Xwindows backend
elif sys.gettrace(): #http://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger
    matplotlib.use('Agg') #because 'QtAgg' crashes python while debugging
else:
    pass
    # matplotlib.use('pdf') #for high quality pdf, but doesn't work for png, svg ...
try:
    logging.info('matplotlib backend is '+matplotlib.get_backend())
except: #fails if matplotlib is a mock for ReadTheDocs
    logging.info('matplotlib is a mock')

class Plot(object):
    """base class for rich object display on IPython notebooks
    inspired from http://nbviewer.ipython.org/github/ipython/ipython/blob/3607712653c66d63e0d7f13f073bde8c0f209ba8/docs/examples/notebooks/display_protocol.ipynb
    """
    
    def _repr_png_(self):
        raise NotImplementedError('no PNG representation defined')

    def _repr_svg_(self):
        raise NotImplementedError('no SVG representation defined')
    
    def _repr_latex_(self):
        raise NotImplementedError('no LaTEX representation defined')
    
    @property
    def png(self):
        from IPython.display import Image
        return Image(self._repr_png_(), embed=True)
    
    @property
    def svg(self):
        from IPython.display import SVG
        return SVG(self._repr_svg_())
    
    @property
    def latex(self):
        from IPython.display import Math
        return Math(self._repr_latex_())