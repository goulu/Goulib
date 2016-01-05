#!/usr/bin/env python
# coding: utf8
"""
hex RGB colors and related functions
"""
__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright 2012-, Philippe Guglielmetti"
__license__ = "LGPL"

import six
from . import math2

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

color = {
    # dict of most used colors indexed by name
    'black': '#000000',
    'blue': '#0000ff',
    'fuchsia': '#ff00ff',
    'green': '#008000',
    'grey': '#808080',
    'lime': '#00ff00',
    'maroon': '#800000',
    'navy': '#000080',
    'olive': '#808000',
    'purple': '#800080',
    'red': '#ff0000',
    'silver': '#c0c0c0',
    'teal': '#008080',
    'white': '#ffffff',
    'yellow': '#ffff00',
    'aliceblue': '#f0f8ff',
    'antiquewhite': '#faebd7',
    'aqua': '#00ffff',
    'aquamarine': '#7fffd4',
    'azure': '#f0ffff',
    'beige': '#f5f5dc',
    'bisque': '#ffe4c4',
    'black': '#000000',
    'blanchedalmond': '#ffebcd',
    'blue': '#0000ff',
    'blueviolet': '#8a2be2',
    'brown': '#a52a2a',
    'burlywood': '#deb887',
    'cadetblue': '#5f9ea0',
    'chartreuse': '#7fff00',
    'chocolate': '#d2691e',
    'coral': '#ff7f50',
    'cornflowerblue': '#6495ed',
    'cornsilk': '#fff8dc',
    'crimson': '#dc143c',
    'cyan': '#00ffff',
    'darkblue': '#00008b',
    'darkcyan': '#008b8b',
    'darkgoldenrod': '#b8860b',
    'darkgray': '#a9a9a9',
    'darkgrey': '#a9a9a9',
    'darkgreen': '#006400',
    'darkkhaki': '#bdb76b',
    'darkmagenta': '#8b008b',
    'darkolivegreen': '#556b2f',
    'darkorange': '#ff8c00',
    'darkorchid': '#9932cc',
    'darkred': '#8b0000',
    'darksalmon': '#e9967a',
    'darkseagreen': '#8fbc8f',
    'darkslateblue': '#483d8b',
    'darkslategray': '#2f4f4f',
    'darkturquoise': '#00ced1',
    'darkviolet': '#9400d3',
    'deeppink': '#ff1493',
    'deepskyblue': '#00bfff',
    'dimgray': '#696969',
    'dimgrey': '#696969',
    'dodgerblue': '#1e90ff',
    'firebrick': '#b22222',
    'floralwhite': '#fffaf0',
    'forestgreen': '#228b22',
    'fuchsia': '#ff00ff',
    'gainsboro': '#dcdcdc',
    'ghostwhite': '#f8f8ff',
    'gold': '#ffd700',
    'goldenrod': '#daa520',
    'gray': '#808080',
    'grey': '#808080',
    'green': '#008000',
    'greenyellow': '#adff2f',
    'honeydew': '#f0fff0',
    'hotpink': '#ff69b4',
    'indianred': '#cd5c5c',
    'indigo': '#4b0082',
    'ivory': '#fffff0',
    'khaki': '#f0e68c',
    'lavender': '#e6e6fa',
    'lavenderblush': '#fff0f5',
    'lawngreen': '#7cfc00',
    'lemonchiffon': '#fffacd',
    'lightblue': '#add8e6',
    'lightcoral': '#f08080',
    'lightcyan': '#e0ffff',
    'lightgoldenrodyellow': '#fafad2',
    'lightgray': '#d3d3d3',
    'lightgrey': '#d3d3d3',
    'lightgreen': '#90ee90',
    'lightpink': '#ffb6c1',
    'lightsalmon': '#ffa07a',
    'lightseagreen': '#20b2aa',
    'lightskyblue': '#87cefa',
    'lightslategray': '#778899',
    'lightslategrey': '#778899',
    'lightsteelblue': '#b0c4de',
    'lightyellow': '#ffffe0',
    'lime': '#00ff00',
    'limegreen': '#32cd32',
    'linen': '#faf0e6',
    'magenta': '#ff00ff',
    'maroon': '#800000',
    'mediumaquamarine': '#66cdaa',
    'mediumblue': '#0000cd',
    'mediumorchid': '#ba55d3',
    'mediumpurple': '#9370d8',
    'mediumseagreen': '#3cb371',
    'mediumslateblue': '#7b68ee',
    'mediumspringgreen': '#00fa9a',
    'mediumturquoise': '#48d1cc',
    'mediumvioletred': '#c71585',
    'midnightblue': '#191970',
    'mintcream': '#f5fffa',
    'mistyrose': '#ffe4e1',
    'moccasin': '#ffe4b5',
    'navajowhite': '#ffdead',
    'navy': '#000080',
    'oldlace': '#fdf5e6',
    'olive': '#808000',
    'olivedrab': '#6b8e23',
    'orange': '#ffa500',
    'orangered': '#ff4500',
    'orchid': '#da70d6',
    'palegoldenrod': '#eee8aa',
    'palegreen': '#98fb98',
    'paleturquoise': '#afeeee',
    'palevioletred': '#d87093',
    'papayawhip': '#ffefd5',
    'peachpuff': '#ffdab9',
    'peru': '#cd853f',
    'pink': '#ffc0cb',
    'plum': '#dda0dd',
    'powderblue': '#b0e0e6',
    'purple': '#800080',
    'red': '#ff0000',
    'rosybrown': '#bc8f8f',
    'royalblue': '#4169e1',
    'saddlebrown': '#8b4513',
    'salmon': '#fa8072',
    'sandybrown': '#f4a460',
    'seagreen': '#2e8b57',
    'seashell': '#fff5ee',
    'sienna': '#a0522d',
    'silver': '#c0c0c0',
    'skyblue': '#87ceeb',
    'slateblue': '#6a5acd',
    'slategray': '#708090',
    'slategrey': '#708090',
    'snow': '#fffafa',
    'springgreen': '#00ff7f',
    'steelblue': '#4682b4',
    'tan': '#d2b48c',
    'teal': '#008080',
    'thistle': '#d8bfd8',
    'tomato': '#ff6347',
    'turquoise': '#40e0d0',
    'violet': '#ee82ee',
    'wheat': '#f5deb3',
    'white': '#ffffff',
    'whitesmoke': '#f5f5f5',
    'yellow': '#ffff00',
    'yellowgreen': '#9acd32'}

color_lookup=dict([v,k] for k,v in color.items()) #http://code.activestate.com/recipes/252143-invert-a-dictionary-one-liner/

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
    def __init__(self,c):
        ''':param c: either color name, hex string, or (r,g,b) tuple in [0..255] int or [0,.1.] float range'''
        if isinstance(c,str):
            try: #is c a color name ?
                c=color[c]
            except: #assume it's a hex string
                pass
            self.rgb=hex_to_rgb(c,1./255)
        else: # assume (r,g,b) tuple
            if max(c)>1:
                c=tuple(_/255. for _ in c)
            self.rgb=c
        
        try: #to guess the color name
            self.name=color_lookup[self.hex]
            return
        except: #find the closest one
            pass
        self.name='~'+nearest_color(self.rgb)
            
    @property
    def hex(self):
        rgb=math2.sat(self.rgb)
        return rgb_to_hex((math2.rint(_*255) for _ in rgb))
    
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
