#!/usr/bin/env python
# coding: utf8
"""
simple color management
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012-, Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ['Colormath http://python-colormath.readthedocs.org/en/latest/',
               'Bruno Nuttens Pantone color table http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/',
               ]

#see http://python-colormath.readthedocs.org/en/latest/ if you need more

import six, os, sys
import numpy as np

from Goulib import math2, itertools2

# color conversion

#redefine some converters in current module to build converters dict below
import skimage.color as skcolor
import matplotlib.colors as mplcolors
rgb2hex=mplcolors.rgb2hex
hex2rgb=mplcolors.hex2color

def rgb2cmyk(rgb):
    """:param r,g,b: floats of red,green,blue in [0..1] range
    :return: tuple of 4 floats (cyan, magenta, yellow, black) in [0..1] range
    """
    # http://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python

    r,g,b=rgb

    c = 1 - r
    m = 1 - g
    y = 1 - b

    k = min(c, m, y)
    if k==1:
        return (0,0,0,1)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return (c,m,y,k)

# skimage.color has several useful color conversion routines, but for images
# so here is a generic adapter that allows to call them with colors

def _skadapt(f):
    def adapted(arr):
        arr = np.asanyarray(arr)
        if arr.ndim ==1:
            res=f(arr.reshape(1,1,arr.shape[-1]))
            return res.reshape(arr.shape[-1])
        else:
            return f(arr)
    return adapted

#supported colorspaces. need more ? just add it :-)
modes=(
    'XYZ',
    'Lab',
    'HSV',
    'CMYK',
    'RGB',
    'HEX',
    #'ACI', #Autocad Color Index [0..255]
    )

#build a graph of available converters
#as in https://github.com/gtaylor/python-colormath

import networkx as nx # http://networkx.github.io/

converters=nx.DiGraph()
for source in modes:
    for target in modes:
        key=(source.lower(),target.lower())
        if key[0]==key[1]:
            continue
        else:
            convname='%s2%s'%key
            converter = getattr(sys.modules[__name__], convname,None)
            if converter is None:
                converter=getattr(skcolor, convname,None)
                if converter: #adapt it:
                    converter=_skadapt(converter)
        if converter:
            converters.add_edge(key[0],key[1],{'f':converter})

def convert(color,source,target):
    """convert a color between colorspaces,
    eventually using intermediary steps
    """
    source,target=source.lower(),target.lower()
    if source==target: return color
    path=nx.shortest_path(converters, source, target)
    for u,v in itertools2.pairwise(path):
        color=converters[u][v]['f'](color)
    return color #isn't it beautiful ?

class Color(object):
    """A color with math operations and conversions
    Color is immutable (as ._values caches representations)
    """
    def __init__(self, value, mode='RGB', name=None):
        """constructor

        :param value: string color name, hex string, or values tuple
        :param mode: string defining the color space of value
        """
        self._name=name
        mode=mode.lower() # for easier conversions

        if isinstance(value,Color): #copy constructor
            self._copy_from_(value)
            return
        if isinstance(value, six.string_types):
            if value in color:
                self._copy_from_(color[value])
                return
            elif value in pantone:
                self._copy_from_(pantone[value])
                return
            elif len(value)==7 and value[0]=='#':
                mode='hex'
            else:
                raise(ValueError("Couldn't create Color(%s,%s)"%(value,mode)))

        self._mode=mode # "native" mode in which the color was created

        if mode=='rgb':
            if max(value)>1:
                value=tuple(_/255. for _ in value)
            value=math2.sat(value,0,1) # do not wash whiter than white...
        if mode!='hex': # force to floats
            value=tuple(float(_) for _ in value)
        self._values={mode:value}


    def _copy_from_(self,c):
        self._mode=c._mode
        self._name=c._name
        self._values=c._values

    @property
    def name(self):
        if self._name is None:
            if self.hex in color_lookup:
                self._name=color_lookup[self.hex].name
            else:
                self._name='~'+nearest_color(self.rgb).name
        return self._name

    def _convert(self, target):
        target=target.lower()
        if target not in self._values:
            path=nx.shortest_path(converters, self._mode, target)
            for u,v in itertools2.pairwise(path):
                if v not in self._values:
                    self._values[v]=converters[u][v]['f'](self._values[u])
        return self._values[target]

    @property
    def rgb(self): return self._convert('rgb')

    @property
    def hex(self): return self._convert('hex')

    @property
    def lab(self): return self._convert('lab')

    @property
    def cmyk(self):return self._convert('cmyk')

    @property
    def hsv(self):return self._convert('hsv')

    def __hash__(self):
        return hash(self.hex)

    def __repr__(self):
        return "Color('%s')"%(self.name)

    def _repr_html_(self):
        return '<p style="color:%s">%s</p>'%(self.hex,self.name)

    def __eq__(self,other):
        try:
            return self.hex==other.hex
        except:
            return self.name==other

    def __add__(self,other):
        return Color(math2.vecadd(self.rgb,other.rgb))

    def __sub__(self,other):
        return Color(math2.vecsub(self.rgb,other.rgb))
    
    def __neg__(self):
        """ complementary color"""
        return color['white']-self

# dictionaries of standardized colors

from Goulib.table import Table

path=os.path.dirname(os.path.abspath(__file__))

# http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/
table=Table(path+'/colors.csv')
table.applyf('hex',lambda x:x.lower())
table=table.groupby('System')

color={} #dict of HTML / matplotlib colors, which seem to be the same
pantone={} #dict of pantone colors

# http://www.w3schools.com/colors/colors_names.asp

for c in table['matplotlib'].asdict():
    id=c['name']
    color[id]=Color(c['hex'],name=id)

color_lookup=dict([v.hex,v] for k,v in color.items()) #http://code.activestate.com/recipes/252143-invert-a-dictionary-one-liner/


for c in table['Pantone'].asdict():
    id=c['name']
    pantone[id]=Color((c['L'],c['a'],c['b']),mode='Lab',name=id)
    # assert p.hex==c['hex'] is always wrong

acadcolors=[None]*256 #table of Autocad indexed colors
for c in table['autocad'].asdict():
    id=c['name']
    acadcolors[id]=Color(c['hex'],name=id) #color name is a 0..255 number


def color_to_aci(x, nearest=True):
    """
    :return: int Autocad Color Index of color x
    """
    if x is None:
        return -1
    x=Color(x)
    try:
        return acadcolors.index(x)
    except:
        pass
    if nearest:
        return nearest_color(x,acadcolors).name # name = int id
    return -1


def aci_to_color(x, block_color=None, layer_color=None):
    if x==0: return block_color
    if x==256: return layer_color
    return acadcolors[x]

def nearest_color(x,l=None):
    """
    :param x: Color
    :param l: list or dict of Color, color_lookup by default
    :return: nearest Color of x in  l
    """
    if not isinstance(x, Color):
        x=Color(x)
    l=l or color_lookup
    if isinstance(l,dict):
        l=l.values()
    return min(l,key=lambda _:math2.dist(x.rgb, _.rgb))

#http://stackoverflow.com/questions/876853/generating-color-ranges-in-python

def color_range(n,start,end):
    """:param n: int number of colors to generate
    :param start: string hex color or color name
    :param end: string hex color or color name
    :result: list of n hexcolors interpolated between start and end, included
    """

    from .itertools2 import linspace
    start=Color(start).hsv
    end=Color(end).hsv
    return [Color(hsv, mode='HSV') for hsv in linspace(start,end,n)]
