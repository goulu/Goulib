#!/usr/bin/env python
# coding: utf8
"""
very simple color management
"""

from __future__ import division #"true division" everywhere

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012-, Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = ['Colormath http://python-colormath.readthedocs.org/en/latest/',
               'Bruno Nuttens Pantone color table http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/',
               ]

#see http://python-colormath.readthedocs.org/en/latest/ if you need more

import six, os 
from Goulib import math2
from Goulib.table import Table

path=os.path.dirname(os.path.abspath(__file__))

# http://blog.brunonuttens.com/206-conversion-couleurs-pantone-lab-rvb-hexa-liste-sql-csv/
table=Table(path+'/colors.csv')
table.applyf('hex',lambda x:x.lower())
table=table.groupby('System')

color=dict([line['name'],line['hex']] for line in table['matplotlib'].asdict())

color_lookup=dict([v,k] for k,v in color.items()) #http://code.activestate.com/recipes/252143-invert-a-dictionary-one-liner/

pantone=dict([line['name'],line['hex']] for line in table['Pantone'].asdict())

# http://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa

def rgb_to_hex(rgb):
    """:param rgb: tuple (r,g,b) of 3 ints 0-255
    :result: string "#rrggbb" in hex suitable for HTML color"""
    return '#%02x%02x%02x' % tuple(rgb)

def hex_to_rgb(value,scale=1):
    """:param value: string "#rrggbb" in hex suitable for HTML color
    :param scale: float optional 1./255 to scale output to [0,1] floats
    :result: tuple (r,g,b) of 3 ints 0-255"""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(scale*int(value[i:i+int(lv/3)], 16) for i in range(0, lv, int(lv/3)))

# http://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python

def rgb_to_cmyk(r,g,b):
    """:param r,g,b: floats of red,green,blue in [0..1] range
    :return: tuple of 4 floats (cyan, magenta, yellow, black) in [0..1] range
    """
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

#http://stackoverflow.com/questions/876853/generating-color-ranges-in-python
    
def color_range(n,start,end):
    """:param n: int number of colors to generate
    :param start: string hex color or color name
    :param end: string hex color or color name
    :result: list of n hexcolors interpolated between start and end, included
    """
    import colorsys
    from .itertools2 import linspace
    if start in color: start=color[start]
    start=hex_to_rgb(start,1./255)
    start=colorsys.rgb_to_hsv(*start)
    if end in color: end=color[end]
    end=hex_to_rgb(end,1./255)
    end=colorsys.rgb_to_hsv(*end)
    res=[]
    for hsv in linspace(start,end,n):
        rgb=colorsys.hsv_to_rgb(*hsv)
        hex=rgb_to_hex(tuple(int(255*x) for x in rgb))
        res.append(hex)
    return res

"""array of 256 Autocad/Autodesk ACI colors produced from http://www.isctex.com/acadcolors.php produced by this code :
    >>> from Goulib.table import Table
    >>> acadcolors=[]
    >>> t=Table('AutoCAD Color Index RGB Equivalents.html')[1:] #outer <table> from original file must be removed
    >>> for c in t: acadcolors.append(rgb_to_hex(c[4:]))
"""
acadcolors=[
    '#000000', '#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#ffffff', 
    '#414141', '#808080', '#ff0000', '#ffaaaa', '#bd0000', '#bd7e7e', '#810000', '#815656', 
    '#680000', '#684545', '#4f0000', '#4f3535', '#ff3f00', '#ffbfaa', '#bd2e00', '#bd8d7e', 
    '#811f00', '#816056', '#681900', '#684e45', '#4f1300', '#4f3b35', '#ff7f00', '#ffd4aa', 
    '#bd5e00', '#bd9d7e', '#814000', '#816b56', '#683400', '#685645', '#4f2700', '#4f4235', 
    '#ffbf00', '#ffeaaa', '#bd8d00', '#bdad7e', '#816000', '#817656', '#684e00', '#685f45', 
    '#4f3b00', '#4f4935', '#ffff00', '#ffffaa', '#bdbd00', '#bdbd7e', '#818100', '#818156', 
    '#686800', '#686845', '#4f4f00', '#4f4f35', '#bfff00', '#eaffaa', '#8dbd00', '#adbd7e', 
    '#608100', '#768156', '#4e6800', '#5f6845', '#3b4f00', '#494f35', '#7fff00', '#d4ffaa', 
    '#5ebd00', '#9dbd7e', '#408100', '#6b8156', '#346800', '#566845', '#274f00', '#424f35', 
    '#3fff00', '#bfffaa', '#2ebd00', '#8dbd7e', '#1f8100', '#608156', '#196800', '#4e6845', 
    '#134f00', '#3b4f35', '#00ff00', '#aaffaa', '#00bd00', '#7ebd7e', '#008100', '#568156', 
    '#006800', '#456845', '#004f00', '#354f35', '#00ff3f', '#aaffbf', '#00bd2e', '#7ebd8d', 
    '#00811f', '#568160', '#006819', '#45684e', '#004f13', '#354f3b', '#00ff7f', '#aaffd4', 
    '#00bd5e', '#7ebd9d', '#008140', '#56816b', '#006834', '#456856', '#004f27', '#354f42', 
    '#00ffbf', '#aaffea', '#00bd8d', '#7ebdad', '#008160', '#568176', '#00684e', '#45685f', 
    '#004f3b', '#354f49', '#00ffff', '#aaffff', '#00bdbd', '#7ebdbd', '#008181', '#568181', 
    '#006868', '#456868', '#004f4f', '#354f4f', '#00bfff', '#aaeaff', '#008dbd', '#7eadbd', 
    '#006081', '#567681', '#004e68', '#455f68', '#003b4f', '#35494f', '#007fff', '#aad4ff', 
    '#005ebd', '#7e9dbd', '#004081', '#566b81', '#003468', '#455668', '#00274f', '#35424f', 
    '#003fff', '#aabfff', '#002ebd', '#7e8dbd', '#001f81', '#566081', '#001968', '#454e68', 
    '#00134f', '#353b4f', '#0000ff', '#aaaaff', '#0000bd', '#7e7ebd', '#000081', '#565681', 
    '#000068', '#454568', '#00004f', '#35354f', '#3f00ff', '#bfaaff', '#2e00bd', '#8d7ebd', 
    '#1f0081', '#605681', '#190068', '#4e4568', '#13004f', '#3b354f', '#7f00ff', '#d4aaff', 
    '#5e00bd', '#9d7ebd', '#400081', '#6b5681', '#340068', '#564568', '#27004f', '#42354f', 
    '#bf00ff', '#eaaaff', '#8d00bd', '#ad7ebd', '#600081', '#765681', '#4e0068', '#5f4568', 
    '#3b004f', '#49354f', '#ff00ff', '#ffaaff', '#bd00bd', '#bd7ebd', '#810081', '#815681', 
    '#680068', '#684568', '#4f004f', '#4f354f', '#ff00bf', '#ffaaea', '#bd008d', '#bd7ead', 
    '#810060', '#815676', '#68004e', '#68455f', '#4f003b', '#4f3549', '#ff007f', '#ffaad4', 
    '#bd005e', '#bd7e9d', '#810040', '#81566b', '#680034', '#684556', '#4f0027', '#4f3542', 
    '#ff003f', '#ffaabf', '#bd002e', '#bd7e8d', '#81001f', '#815660', '#680019', '#68454e', 
    '#4f0013', '#4f353b', '#333333', '#505050', '#696969', '#828282', '#bebebe', '#ffffff']


def color_to_aci(x, nearest=True):
    """
    :return: int Autocad Color Index of color x
    """
    if x is None:
        return -1
    if x in color :  #color name
        x=color[x]
    try:
        return acadcolors.index(x)
    except:
        pass
    if nearest:
        return _nearest(x,acadcolors)[0] #return index only
    else:
        return acadcolors.index(color_lookup[x])

    
def aci_to_color(x, block_color=None, layer_color=None):
    if x==0: return block_color
    if x==256: return layer_color
    c=acadcolors[x]
    try: #handle standard Matplotlib colors by name
        c=color_lookup[c]
    except:
        pass
    return c

def _nearest(x,l):
    """:return: index  of the nearest color in list l"""
    if isinstance(x,six.string_types):
        rgb=hex_to_rgb(x,1./255)
    else:
        rgb=math2.sat(x,0,1)
    from .itertools2 import index_min
    return index_min(l,key=lambda _:math2.dist(rgb, hex_to_rgb(_,1./255)))

def nearest_color(x,l=None):
    """:return: name of the nearest color in list l or in color_lootup table"""
    l=l or color_lookup
    i,c=_nearest(x,l)
    if isinstance(l,dict):
        return l[c]
    return l[i]

class Color(object):
    '''class to allow simple math operations on colors'''
    def __init__(self, name, rgb=None, lab=None):
        """constructor
        
        :param name: string color name, hex string, or (r,g,b) tuple in [0..255] int or [0,.1.] float range
        """
        
        def init(name=None, rgb=None, hex=None, lab=None):
            self._name=name
            if rgb:
                if max(rgb)>1:
                    rgb=tuple(_/255. for _ in rgb)
                rgb=math2.sat(rgb,0,1) #nothing is whiter than white...
            self._rgb=rgb
            self._hex=hex
            self._lab=lab
            
        if isinstance(name,Color): #copy constructor
            init(name.name,name.rgb,name.hex,name.lab)
        elif isinstance(name, six.string_types):
            if name in color:
                init(name,hex=color[name])
            elif name in pantone:
                init(name,hex=pantone[name]) #TODO: use Lab instead
            else:
                init(hex=name)
        elif lab is not None:
            init(lab=lab)
        else:  # assume (r,g,b) tuple
            init(rgb=rgb or name)

        
    @property
    def name(self):
        if self._name is None:
            if self.hex in color_lookup:
                self._name=color_lookup[self.hex]
            else:
                self._name='~'+nearest_color(self.rgb)
        return self._name
            
    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb=hex_to_rgb(self._hex, 1/255)     
        return self._rgb 
    
    @property
    def hex(self):
        if self._hex is None:
            self._hex=rgb_to_hex((math2.rint(_*255) for _ in self.rgb))
        return self._hex
    
    @property
    def lab(self):
        if self._lab is None:
            pass #TODO: implement
        return self._lab
    
    def __repr__(self):
        return "Color('%s')"%(self.name if self.name[0]!='~' else self.hex)
        
    def _repr_html_(self):
        return '<div style="color:%s">%s</div>'%(self.hex,self.name)
        
    def __eq__(self,other):
        try:
            return self.hex==other.hex
        except:
            return self.name==other
    
    def __add__(self,other):
        return Color(math2.vecadd(self.rgb,other.rgb))
    
    def __sub__(self,other):        
        return Color(math2.vecsub(self.rgb,other.rgb))
